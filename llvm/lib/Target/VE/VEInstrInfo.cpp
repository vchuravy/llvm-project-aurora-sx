//===-- VEInstrInfo.cpp - VE Instruction Information ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the VE implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "VEInstrInfo.h"
#include "VE.h"
#include "VEMachineFunctionInfo.h"
#include "VESubtarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define DEBUG_TYPE "ve-instr-info"

using namespace llvm;

static cl::opt<bool> ShowSpillMessageVec(
  "show-spill-message-vec",
  cl::init(false),
  cl::desc("Enable diagnostic message for spill/restore of vector or vector mask registers."),
  cl::Hidden);

#define GET_INSTRINFO_CTOR_DTOR
#include "VEGenInstrInfo.inc"

// Pin the vtable to this file.
void VEInstrInfo::anchor() {}

VEInstrInfo::VEInstrInfo(VESubtarget &ST)
    : VEGenInstrInfo(VE::ADJCALLSTACKDOWN, VE::ADJCALLSTACKUP), RI(),
      Subtarget(ST) {}

static bool IsIntegerCC(unsigned CC) { return (CC < VECC::CC_AF); }

static VECC::CondCode GetOppositeBranchCondition(VECC::CondCode CC) {
  switch (CC) {
  case VECC::CC_IG:
    return VECC::CC_ILE;
  case VECC::CC_IL:
    return VECC::CC_IGE;
  case VECC::CC_INE:
    return VECC::CC_IEQ;
  case VECC::CC_IEQ:
    return VECC::CC_INE;
  case VECC::CC_IGE:
    return VECC::CC_IL;
  case VECC::CC_ILE:
    return VECC::CC_IG;
  case VECC::CC_AF:
    return VECC::CC_AT;
  case VECC::CC_G:
    return VECC::CC_LENAN;
  case VECC::CC_L:
    return VECC::CC_GENAN;
  case VECC::CC_NE:
    return VECC::CC_EQNAN;
  case VECC::CC_EQ:
    return VECC::CC_NENAN;
  case VECC::CC_GE:
    return VECC::CC_LNAN;
  case VECC::CC_LE:
    return VECC::CC_GNAN;
  case VECC::CC_NUM:
    return VECC::CC_NAN;
  case VECC::CC_NAN:
    return VECC::CC_NUM;
  case VECC::CC_GNAN:
    return VECC::CC_LE;
  case VECC::CC_LNAN:
    return VECC::CC_GE;
  case VECC::CC_NENAN:
    return VECC::CC_EQ;
  case VECC::CC_EQNAN:
    return VECC::CC_NE;
  case VECC::CC_GENAN:
    return VECC::CC_L;
  case VECC::CC_LENAN:
    return VECC::CC_G;
  case VECC::CC_AT:
    return VECC::CC_AF;
  case VECC::UNKNOWN:
    return VECC::UNKNOWN;
  }
  llvm_unreachable("Invalid cond code");
}

// Treat branch relative always like br.l.t as unconditional branch
// instructions.
static bool isUncondBranchOpcode(int Opc) {
  using namespace llvm::VE;

#define BRKIND(NAME) \
    (Opc == NAME ## a || Opc == NAME ## a_nt || Opc == NAME ## a_t)
  return BRKIND(BRCFL) || BRKIND(BRCFW) || BRKIND(BRCFD) || BRKIND(BRCFS);
#undef BRKIND
}

// Treat branch relative conditional like brgt.l.t as conditional branch
// instructions.
static bool isCondBranchOpcode(int Opc) {
  using namespace llvm::VE;

#define BRKIND(NAME) \
    (Opc == NAME ## rr || Opc == NAME ## rr_nt || Opc == NAME ## rr_t || \
     Opc == NAME ## ir || Opc == NAME ## ir_nt || Opc == NAME ## ir_t)
  return BRKIND(BRCFL) || BRKIND(BRCFW) || BRKIND(BRCFD) || BRKIND(BRCFS);
#undef BRKIND
}

// Treat branch always like b.l.t as indirect branch instructions.
static bool isIndirectBranchOpcode(int Opc) {
  using namespace llvm::VE;

#define BRKIND(NAME) \
    (Opc == NAME ## ari || Opc == NAME ## ari_nt || Opc == NAME ## ari_t)
  return BRKIND(BCFL) || BRKIND(BCFW) || BRKIND(BCFD) || BRKIND(BCFS);
#undef BRKIND
}

static void parseCondBranch(MachineInstr *LastInst, MachineBasicBlock *&Target,
                            SmallVectorImpl<MachineOperand> &Cond) {
  Cond.push_back(MachineOperand::CreateImm(LastInst->getOperand(0).getImm()));
  Cond.push_back(LastInst->getOperand(1));
  Cond.push_back(LastInst->getOperand(2));
  Target = LastInst->getOperand(3).getMBB();
}

bool VEInstrInfo::analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                                MachineBasicBlock *&FBB,
                                SmallVectorImpl<MachineOperand> &Cond,
                                bool AllowModify) const {
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end())
    return false;

  if (!isUnpredicatedTerminator(*I))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = &*I;
  unsigned LastOpc = LastInst->getOpcode();

  // If there is only one terminator instruction, process it.
  if (I == MBB.begin() || !isUnpredicatedTerminator(*--I)) {
    if (isUncondBranchOpcode(LastOpc)) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }
    if (isCondBranchOpcode(LastOpc)) {
      // Block ends with fall-through condbranch.
      parseCondBranch(LastInst, TBB, Cond);
      return false;
    }
    return true; // Can't handle indirect branch.
  }

  // Get the instruction before it if it is a terminator.
  MachineInstr *SecondLastInst = &*I;
  unsigned SecondLastOpc = SecondLastInst->getOpcode();

  // If AllowModify is true and the block ends with two or more unconditional
  // branches, delete all but the first unconditional branch.
  if (AllowModify && isUncondBranchOpcode(LastOpc)) {
    while (isUncondBranchOpcode(SecondLastOpc)) {
      LastInst->eraseFromParent();
      LastInst = SecondLastInst;
      LastOpc = LastInst->getOpcode();
      if (I == MBB.begin() || !isUnpredicatedTerminator(*--I)) {
        // Return now the only terminator is an unconditional branch.
        TBB = LastInst->getOperand(0).getMBB();
        return false;
      }
      SecondLastInst = &*I;
      SecondLastOpc = SecondLastInst->getOpcode();
    }
  }

  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() && isUnpredicatedTerminator(*--I))
    return true;

  // If the block ends with a B and a Bcc, handle it.
  if (isCondBranchOpcode(SecondLastOpc) && isUncondBranchOpcode(LastOpc)) {
    parseCondBranch(SecondLastInst, TBB, Cond);
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }

  // If the block ends with two unconditional branches, handle it.  The second
  // one is not executed.
  if (isUncondBranchOpcode(SecondLastOpc) && isUncondBranchOpcode(LastOpc)) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    return false;
  }

  // ...likewise if it ends with an indirect branch followed by an unconditional
  // branch.
  if (isIndirectBranchOpcode(SecondLastOpc) && isUncondBranchOpcode(LastOpc)) {
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return true;
  }

  // Otherwise, can't handle this.
  return true;
}

unsigned VEInstrInfo::insertBranch(MachineBasicBlock &MBB,
                                   MachineBasicBlock *TBB,
                                   MachineBasicBlock *FBB,
                                   ArrayRef<MachineOperand> Cond,
                                   const DebugLoc &DL, int *BytesAdded) const {
  assert(TBB && "insertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 3 || Cond.size() == 0) &&
         "VE branch conditions should have three component!");
  assert(!BytesAdded && "code size not handled");
  if (Cond.empty()) {
    // Uncondition branch
    assert(!FBB && "Unconditional branch with multiple successors!");
    BuildMI(&MBB, DL, get(VE::BRCFLa_t))
        .addMBB(TBB);
    return 1;
  }

  // Conditional branch
  //   (BRCFir CC sy sz addr)
  assert(Cond[0].isImm() && Cond[2].isReg() && "not implemented");

  unsigned opc[2];
  const TargetRegisterInfo *TRI = &getRegisterInfo();
  MachineFunction *MF = MBB.getParent();
  const MachineRegisterInfo &MRI = MF->getRegInfo();
  unsigned Reg = Cond[2].getReg();
  if (IsIntegerCC(Cond[0].getImm())) {
    if (TRI->getRegSizeInBits(Reg, MRI) == 32) {
      opc[0] = VE::BRCFWir;
      opc[1] = VE::BRCFWrr;
    } else {
      opc[0] = VE::BRCFLir;
      opc[1] = VE::BRCFLrr;
    }
  } else {
    if (TRI->getRegSizeInBits(Reg, MRI) == 32) {
      opc[0] = VE::BRCFSir;
      opc[1] = VE::BRCFSrr;
    } else {
      opc[0] = VE::BRCFDir;
      opc[1] = VE::BRCFDrr;
    }
  }
  if (Cond[1].isImm()) {
      BuildMI(&MBB, DL, get(opc[0]))
          .add(Cond[0]) // condition code
          .add(Cond[1]) // lhs
          .add(Cond[2]) // rhs
          .addMBB(TBB);
  } else {
      BuildMI(&MBB, DL, get(opc[1]))
          .add(Cond[0])
          .add(Cond[1])
          .add(Cond[2])
          .addMBB(TBB);
  }

  if (!FBB)
    return 1;

  BuildMI(&MBB, DL, get(VE::BRCFLa_t))
      .addMBB(FBB);
  return 2;
}

unsigned VEInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                   int *BytesRemoved) const {
  assert(!BytesRemoved && "code size not handled");

  MachineBasicBlock::iterator I = MBB.end();
  unsigned Count = 0;
  while (I != MBB.begin()) {
    --I;

    if (I->isDebugValue())
      continue;

    if (!isUncondBranchOpcode(I->getOpcode()) &&
        !isCondBranchOpcode(I->getOpcode()))
      break; // Not a branch

    I->eraseFromParent();
    I = MBB.end();
    ++Count;
  }
  return Count;
}

bool VEInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  VECC::CondCode CC = static_cast<VECC::CondCode>(Cond[0].getImm());
  Cond[0].setImm(GetOppositeBranchCondition(CC));
  return false;
}

void VEInstrInfo::copyPhysSubRegs(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I,
                                  const DebugLoc &DL, MCRegister DestReg,
                                  MCRegister SrcReg, bool KillSrc,
                                  const MCInstrDesc &MCID,
                                  unsigned int numSubRegs,
                                  const unsigned* subRegIdx) const {
  const TargetRegisterInfo *TRI = &getRegisterInfo();
  MachineInstr *MovMI = nullptr;

  for (unsigned i = 0; i != numSubRegs; ++i) {
    unsigned SubDest = TRI->getSubReg(DestReg, subRegIdx[i]);
    unsigned SubSrc  = TRI->getSubReg(SrcReg,  subRegIdx[i]);
    assert(SubDest && SubSrc && "Bad sub-register");

    if (MCID.getOpcode() == VE::ORri) {
      // generate "ORri, dest, src, 0" instruction.
      MachineInstrBuilder MIB = BuildMI(MBB, I, DL, MCID, SubDest)
          .addReg(SubSrc).addImm(0);
      MovMI = MIB.getInstr();
    } else if (MCID.getOpcode() == VE::andm_mmm) {
      // generate "ANDM, dest, vm0, src" instruction.
      MachineInstrBuilder MIB = BuildMI(MBB, I, DL, MCID, SubDest)
          .addReg(VE::VM0).addReg(SubSrc);
      MovMI = MIB.getInstr();
    } else {
      llvm_unreachable("Unexpected reg-to-reg copy instruction");
    }
  }
  // Add implicit super-register defs and kills to the last MovMI.
  MovMI->addRegisterDefined(DestReg, TRI);
  if (KillSrc)
    MovMI->addRegisterKilled(SrcReg, TRI, true);
}

static bool IsAliasOfSX(Register Reg) {
  return VE::I32RegClass.contains(Reg) || VE::I64RegClass.contains(Reg) ||
         VE::F32RegClass.contains(Reg);
}

void VEInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I, const DebugLoc &DL,
                              MCRegister DestReg, MCRegister SrcReg,
                              bool KillSrc) const {

  if (IsAliasOfSX(SrcReg) && IsAliasOfSX(DestReg)) {
    BuildMI(MBB, I, DL, get(VE::ORri), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addImm(0);
  } else if (VE::V64RegClass.contains(DestReg, SrcReg)) {
    // Generate following instructions
    //   %sw16 = LEA32zii 256
    //   vor_v1vl %dest, (0)1, %src, %sw16
    // TODO: reuse a register if vl is already assigned to a register
    // FIXME: it would be better to scavenge a register here instead of
    // reserving SX16 all of the time.
    const TargetRegisterInfo *TRI = &getRegisterInfo();
    unsigned TmpReg = VE::SX16;
    unsigned SubTmp = TRI->getSubReg(TmpReg, VE::sub_i32);
    BuildMI(MBB, I, DL, get(VE::LEAzii), TmpReg)
        .addImm(0).addImm(0).addImm(256);
    MachineInstrBuilder MIB = BuildMI(MBB, I, DL, get(VE::vor_v1vl), DestReg)
        .addImm(0)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addReg(SubTmp, getKillRegState(true));
    MIB.getInstr()->addRegisterKilled(TmpReg, TRI, true);
  } else if (VE::VMRegClass.contains(DestReg, SrcReg))
    BuildMI(MBB, I, DL, get(VE::andm_mmm), DestReg)
        .addReg(VE::VM0)
        .addReg(SrcReg, getKillRegState(KillSrc));
  else if (VE::VM512RegClass.contains(DestReg, SrcReg)) {
    // Use two instructions.
    const unsigned subRegIdx[] = { VE::sub_vm_even, VE::sub_vm_odd };
    unsigned int numSubRegs = 2;
    copyPhysSubRegs(MBB, I, DL, DestReg, SrcReg, KillSrc, get(VE::andm_mmm),
                    numSubRegs, subRegIdx);
  } else if (VE::F128RegClass.contains(DestReg, SrcReg)) {
    // Use two instructions.
    const unsigned subRegIdx[] = { VE::sub_even, VE::sub_odd };
    unsigned int numSubRegs = 2;
    copyPhysSubRegs(MBB, I, DL, DestReg, SrcReg, KillSrc, get(VE::ORri),
                    numSubRegs, subRegIdx);
  } else {
    const TargetRegisterInfo *TRI = &getRegisterInfo();
    dbgs() << "Impossible reg-to-reg copy from " << printReg(SrcReg, TRI)
           << " to " << printReg(DestReg, TRI) << "\n";
    llvm_unreachable("Impossible reg-to-reg copy");
  }
}

/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned VEInstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                          int &FrameIndex) const {
  if (MI.getOpcode() == VE::LDrii ||            // I64
      MI.getOpcode() == VE::LDLSXrii ||         // I32
      MI.getOpcode() == VE::LDUrii ||           // F32
      MI.getOpcode() == VE::LDQrii ||           // F128 (pseudo)
      MI.getOpcode() == VE::LDVRrii ||          // V64 (pseudo)
      MI.getOpcode() == VE::LDVMrii ||          // VM (pseudo)
      MI.getOpcode() == VE::LDVM512rii          // VM512 (pseudo)
  ) {
    if (MI.getOperand(1).isFI() && MI.getOperand(2).isImm() &&
        MI.getOperand(2).getImm() == 0 && MI.getOperand(3).isImm() &&
        MI.getOperand(3).getImm() == 0) {
      FrameIndex = MI.getOperand(1).getIndex();
      return MI.getOperand(0).getReg();
    }
  }
  return 0;
}

/// isStoreToStackSlot - If the specified machine instruction is a direct
/// store to a stack slot, return the virtual or physical register number of
/// the source reg along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than storing to the stack slot.
unsigned VEInstrInfo::isStoreToStackSlot(const MachineInstr &MI,
                                         int &FrameIndex) const {
  if (MI.getOpcode() == VE::STrii ||            // I64
      MI.getOpcode() == VE::STLrii ||           // I32
      MI.getOpcode() == VE::STUrii ||           // F32
      MI.getOpcode() == VE::STQrii ||           // F128 (pseudo)
      MI.getOpcode() == VE::STVRrii ||          // V64 (pseudo)
      MI.getOpcode() == VE::STVMrii ||          // VM (pseudo)
      MI.getOpcode() == VE::STVM512rii          // VM512 (pseudo)
  ) {
    if (MI.getOperand(0).isFI() && MI.getOperand(1).isImm() &&
        MI.getOperand(1).getImm() == 0 && MI.getOperand(2).isImm() &&
        MI.getOperand(2).getImm() == 0) {
      FrameIndex = MI.getOperand(0).getIndex();
      return MI.getOperand(3).getReg();
    }
  }
  return 0;
}

void VEInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator I,
                                      Register SrcReg, bool isKill, int FI,
                                      const TargetRegisterClass *RC,
                                      const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (I != MBB.end())
    DL = I->getDebugLoc();

  if (ShowSpillMessageVec) {
    if (RC == &VE::V64RegClass) {
      dbgs() << "spill " << printReg(SrcReg, TRI) << " - V64\n";
    } else if (RC == &VE::VMRegClass) {
      dbgs() << "spill " << printReg(SrcReg, TRI) << " - VM\n";
    } else if (VE::VM512RegClass.hasSubClassEq(RC)) {
      dbgs() << "spill " << printReg(SrcReg, TRI) << " - VM512\n";
    }
  }

  MachineFunction *MF = MBB.getParent();
  const MachineFrameInfo &MFI = MF->getFrameInfo();
  MachineMemOperand *MMO = MF->getMachineMemOperand(
      MachinePointerInfo::getFixedStack(*MF, FI), MachineMemOperand::MOStore,
      MFI.getObjectSize(FI), MFI.getObjectAlign(FI));

  // On the order of operands here: think "[FrameIdx + 0] = SrcReg".
  if (RC == &VE::I64RegClass) {
    BuildMI(MBB, I, DL, get(VE::STrii))
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addReg(SrcReg, getKillRegState(isKill))
        .addMemOperand(MMO);
  } else if (RC == &VE::I32RegClass) {
    BuildMI(MBB, I, DL, get(VE::STLrii))
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addReg(SrcReg, getKillRegState(isKill))
        .addMemOperand(MMO);
  } else if (RC == &VE::F32RegClass) {
    BuildMI(MBB, I, DL, get(VE::STUrii))
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addReg(SrcReg, getKillRegState(isKill))
        .addMemOperand(MMO);
  } else if (VE::F128RegClass.hasSubClassEq(RC)) {
    BuildMI(MBB, I, DL, get(VE::STQrii))
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addReg(SrcReg, getKillRegState(isKill))
        .addMemOperand(MMO);
  } else if (RC == &VE::V64RegClass) {
    BuildMI(MBB, I, DL, get(VE::STVRrii))
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addReg(SrcReg, getKillRegState(isKill))
        .addImm(256)
        .addMemOperand(MMO);
  } else if (RC == &VE::VMRegClass) {
    BuildMI(MBB, I, DL, get(VE::STVMrii))
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addReg(SrcReg, getKillRegState(isKill))
        .addMemOperand(MMO);
  } else if (VE::VM512RegClass.hasSubClassEq(RC)) {
    BuildMI(MBB, I, DL, get(VE::STVM512rii))
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addReg(SrcReg, getKillRegState(isKill))
        .addMemOperand(MMO);
  } else
    report_fatal_error("Can't store this register to stack slot");
}

void VEInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I,
                                       Register DestReg, int FI,
                                       const TargetRegisterClass *RC,
                                       const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (I != MBB.end())
    DL = I->getDebugLoc();

  if (ShowSpillMessageVec) {
    if (RC == &VE::V64RegClass) {
      dbgs() << "restore " << printReg(DestReg, TRI) << " - V64\n";
    } else if (RC == &VE::VMRegClass) {
      dbgs() << "restore " << printReg(DestReg, TRI) << " - VM\n";
    } else if (VE::VM512RegClass.hasSubClassEq(RC)) {
      dbgs() << "restore " << printReg(DestReg, TRI) << " - VM512\n";
    }
  }

  MachineFunction *MF = MBB.getParent();
  const MachineFrameInfo &MFI = MF->getFrameInfo();
  MachineMemOperand *MMO = MF->getMachineMemOperand(
      MachinePointerInfo::getFixedStack(*MF, FI), MachineMemOperand::MOLoad,
      MFI.getObjectSize(FI), MFI.getObjectAlign(FI));

  if (RC == &VE::I64RegClass) {
    BuildMI(MBB, I, DL, get(VE::LDrii), DestReg)
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addMemOperand(MMO);
  } else if (RC == &VE::I32RegClass) {
    BuildMI(MBB, I, DL, get(VE::LDLSXrii), DestReg)
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addMemOperand(MMO);
  } else if (RC == &VE::F32RegClass) {
    BuildMI(MBB, I, DL, get(VE::LDUrii), DestReg)
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addMemOperand(MMO);
  } else if (VE::F128RegClass.hasSubClassEq(RC)) {
    BuildMI(MBB, I, DL, get(VE::LDQrii), DestReg)
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addMemOperand(MMO);
  } else if (RC == &VE::V64RegClass) {
    BuildMI(MBB, I, DL, get(VE::LDVRrii), DestReg)
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addImm(256)
        .addMemOperand(MMO);
  } else if (RC == &VE::VMRegClass) {
    BuildMI(MBB, I, DL, get(VE::LDVMrii), DestReg)
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addMemOperand(MMO);
  } else if (VE::VM512RegClass.hasSubClassEq(RC)) {
    BuildMI(MBB, I, DL, get(VE::LDVM512rii), DestReg)
        .addFrameIndex(FI)
        .addImm(0)
        .addImm(0)
        .addMemOperand(MMO);
  } else
    report_fatal_error("Can't load this register from stack slot");
}

Register VEInstrInfo::getGlobalBaseReg(MachineFunction *MF) const {
  VEMachineFunctionInfo *VEFI = MF->getInfo<VEMachineFunctionInfo>();
  Register GlobalBaseReg = VEFI->getGlobalBaseReg();
  if (GlobalBaseReg != 0)
    return GlobalBaseReg;

  // We use %s15 (%got) as a global base register
  GlobalBaseReg = VE::SX15;

  // Insert a pseudo instruction to set the GlobalBaseReg into the first
  // MBB of the function
  MachineBasicBlock &FirstMBB = MF->front();
  MachineBasicBlock::iterator MBBI = FirstMBB.begin();
  DebugLoc dl;
  BuildMI(FirstMBB, MBBI, dl, get(VE::GETGOT), GlobalBaseReg);
  VEFI->setGlobalBaseReg(GlobalBaseReg);
  return GlobalBaseReg;
}

static int GetVM512Upper(int no)
{
    return (no - VE::VMP0) * 2 + VE::VM0;
}

static int GetVM512Lower(int no)
{
    return GetVM512Upper(no) + 1;
}

static void buildVMRInst(MachineInstr& MI, const MCInstrDesc& MCID) {
  MachineBasicBlock* MBB = MI.getParent();
  DebugLoc dl = MI.getDebugLoc();

  unsigned VMXu = GetVM512Upper(MI.getOperand(0).getReg());
  unsigned VMXl = GetVM512Lower(MI.getOperand(0).getReg());
  unsigned VMYu = GetVM512Upper(MI.getOperand(1).getReg());
  unsigned VMYl = GetVM512Lower(MI.getOperand(1).getReg());

  switch (MI.getOpcode()) {
  default: {
      unsigned VMZu = GetVM512Upper(MI.getOperand(2).getReg());
      unsigned VMZl = GetVM512Lower(MI.getOperand(2).getReg());
      BuildMI(*MBB, MI, dl, MCID).addDef(VMXu).addUse(VMYu).addUse(VMZu);
      BuildMI(*MBB, MI, dl, MCID).addDef(VMXl).addUse(VMYl).addUse(VMZl);
      break;
  }
  case VE::negm_MM:
      BuildMI(*MBB, MI, dl, MCID).addDef(VMXu).addUse(VMYu);
      BuildMI(*MBB, MI, dl, MCID).addDef(VMXl).addUse(VMYl);
      break;
  }
  MI.eraseFromParent();
}

static void expandPseudoVFMK_VL(const TargetInstrInfo& TI, MachineInstr& MI)
{
    // replace to pvfmk.w.up and pvfmk.w.lo
    // replace to pvfmk.s.up and pvfmk.s.lo

    std::map<int, std::vector<int>> map = {
      {VE::pvfmkat_Ml, {VE::pvfmkwupat_ml, VE::pvfmkwloat_ml}},
      {VE::pvfmkaf_Ml, {VE::pvfmkwupaf_ml, VE::pvfmkwloaf_ml}},
      {VE::pvfmkwgt_Mvl, {VE::pvfmkwupgt_mvl, VE::pvfmkwlogt_mvl}},
      {VE::pvfmkwlt_Mvl, {VE::pvfmkwuplt_mvl, VE::pvfmkwlolt_mvl}},
      {VE::pvfmkwne_Mvl, {VE::pvfmkwupne_mvl, VE::pvfmkwlone_mvl}},
      {VE::pvfmkweq_Mvl, {VE::pvfmkwupeq_mvl, VE::pvfmkwloeq_mvl}},
      {VE::pvfmkwge_Mvl, {VE::pvfmkwupge_mvl, VE::pvfmkwloge_mvl}},
      {VE::pvfmkwle_Mvl, {VE::pvfmkwuple_mvl, VE::pvfmkwlole_mvl}},
      {VE::pvfmkwnum_Mvl, {VE::pvfmkwupnum_mvl, VE::pvfmkwlonum_mvl}},
      {VE::pvfmkwnan_Mvl, {VE::pvfmkwupnan_mvl, VE::pvfmkwlonan_mvl}},
      {VE::pvfmkwgtnan_Mvl, {VE::pvfmkwupgtnan_mvl, VE::pvfmkwlogtnan_mvl}},
      {VE::pvfmkwltnan_Mvl, {VE::pvfmkwupltnan_mvl, VE::pvfmkwloltnan_mvl}},
      {VE::pvfmkwnenan_Mvl, {VE::pvfmkwupnenan_mvl, VE::pvfmkwlonenan_mvl}},
      {VE::pvfmkweqnan_Mvl, {VE::pvfmkwupeqnan_mvl, VE::pvfmkwloeqnan_mvl}},
      {VE::pvfmkwgenan_Mvl, {VE::pvfmkwupgenan_mvl, VE::pvfmkwlogenan_mvl}},
      {VE::pvfmkwlenan_Mvl, {VE::pvfmkwuplenan_mvl, VE::pvfmkwlolenan_mvl}},

      {VE::pvfmkwgt_MvMl, {VE::pvfmkwupgt_mvml, VE::pvfmkwlogt_mvml}},
      {VE::pvfmkwlt_MvMl, {VE::pvfmkwuplt_mvml, VE::pvfmkwlolt_mvml}},
      {VE::pvfmkwne_MvMl, {VE::pvfmkwupne_mvml, VE::pvfmkwlone_mvml}},
      {VE::pvfmkweq_MvMl, {VE::pvfmkwupeq_mvml, VE::pvfmkwloeq_mvml}},
      {VE::pvfmkwge_MvMl, {VE::pvfmkwupge_mvml, VE::pvfmkwloge_mvml}},
      {VE::pvfmkwle_MvMl, {VE::pvfmkwuple_mvml, VE::pvfmkwlole_mvml}},
      {VE::pvfmkwnum_MvMl, {VE::pvfmkwupnum_mvml, VE::pvfmkwlonum_mvml}},
      {VE::pvfmkwnan_MvMl, {VE::pvfmkwupnan_mvml, VE::pvfmkwlonan_mvml}},
      {VE::pvfmkwgtnan_MvMl, {VE::pvfmkwupgtnan_mvml, VE::pvfmkwlogtnan_mvml}},
      {VE::pvfmkwltnan_MvMl, {VE::pvfmkwupltnan_mvml, VE::pvfmkwloltnan_mvml}},
      {VE::pvfmkwnenan_MvMl, {VE::pvfmkwupnenan_mvml, VE::pvfmkwlonenan_mvml}},
      {VE::pvfmkweqnan_MvMl, {VE::pvfmkwupeqnan_mvml, VE::pvfmkwloeqnan_mvml}},
      {VE::pvfmkwgenan_MvMl, {VE::pvfmkwupgenan_mvml, VE::pvfmkwlogenan_mvml}},
      {VE::pvfmkwlenan_MvMl, {VE::pvfmkwuplenan_mvml, VE::pvfmkwlolenan_mvml}},

      {VE::pvfmksgt_Mvl, {VE::pvfmksupgt_mvl, VE::pvfmkslogt_mvl}},
      {VE::pvfmksgt_MvMl, {VE::pvfmksupgt_mvml, VE::pvfmkslogt_mvml}},
      {VE::pvfmkslt_Mvl, {VE::pvfmksuplt_mvl, VE::pvfmkslolt_mvl}},
      {VE::pvfmksne_Mvl, {VE::pvfmksupne_mvl, VE::pvfmkslone_mvl}},
      {VE::pvfmkseq_Mvl, {VE::pvfmksupeq_mvl, VE::pvfmksloeq_mvl}},
      {VE::pvfmksge_Mvl, {VE::pvfmksupge_mvl, VE::pvfmksloge_mvl}},
      {VE::pvfmksle_Mvl, {VE::pvfmksuple_mvl, VE::pvfmkslole_mvl}},
      {VE::pvfmksnum_Mvl, {VE::pvfmksupnum_mvl, VE::pvfmkslonum_mvl}},
      {VE::pvfmksnan_Mvl, {VE::pvfmksupnan_mvl, VE::pvfmkslonan_mvl}},
      {VE::pvfmksgtnan_Mvl, {VE::pvfmksupgtnan_mvl, VE::pvfmkslogtnan_mvl}},
      {VE::pvfmksltnan_Mvl, {VE::pvfmksupltnan_mvl, VE::pvfmksloltnan_mvl}},
      {VE::pvfmksnenan_Mvl, {VE::pvfmksupnenan_mvl, VE::pvfmkslonenan_mvl}},
      {VE::pvfmkseqnan_Mvl, {VE::pvfmksupeqnan_mvl, VE::pvfmksloeqnan_mvl}},
      {VE::pvfmksgenan_Mvl, {VE::pvfmksupgenan_mvl, VE::pvfmkslogenan_mvl}},
      {VE::pvfmkslenan_Mvl, {VE::pvfmksuplenan_mvl, VE::pvfmkslolenan_mvl}},

      {VE::pvfmksgt_MvMl, {VE::pvfmksupgt_mvml, VE::pvfmkslogt_mvml}},
      {VE::pvfmkslt_MvMl, {VE::pvfmksuplt_mvml, VE::pvfmkslolt_mvml}},
      {VE::pvfmksne_MvMl, {VE::pvfmksupne_mvml, VE::pvfmkslone_mvml}},
      {VE::pvfmkseq_MvMl, {VE::pvfmksupeq_mvml, VE::pvfmksloeq_mvml}},
      {VE::pvfmksge_MvMl, {VE::pvfmksupge_mvml, VE::pvfmksloge_mvml}},
      {VE::pvfmksle_MvMl, {VE::pvfmksuple_mvml, VE::pvfmkslole_mvml}},
      {VE::pvfmksnum_MvMl, {VE::pvfmksupnum_mvml, VE::pvfmkslonum_mvml}},
      {VE::pvfmksnan_MvMl, {VE::pvfmksupnan_mvml, VE::pvfmkslonan_mvml}},
      {VE::pvfmksgtnan_MvMl, {VE::pvfmksupgtnan_mvml, VE::pvfmkslogtnan_mvml}},
      {VE::pvfmksltnan_MvMl, {VE::pvfmksupltnan_mvml, VE::pvfmksloltnan_mvml}},
      {VE::pvfmksnenan_MvMl, {VE::pvfmksupnenan_mvml, VE::pvfmkslonenan_mvml}},
      {VE::pvfmkseqnan_MvMl, {VE::pvfmksupeqnan_mvml, VE::pvfmksloeqnan_mvml}},
      {VE::pvfmksgenan_MvMl, {VE::pvfmksupgenan_mvml, VE::pvfmkslogenan_mvml}},
      {VE::pvfmkslenan_MvMl, {VE::pvfmksuplenan_mvml, VE::pvfmkslolenan_mvml}},
    };

    unsigned Opcode = MI.getOpcode();

    if (map.find(Opcode) == map.end()) {
      report_fatal_error("unexpected opcode for pvfmk");
    }

    unsigned OpcodeUpper = map[Opcode][0];
    unsigned OpcodeLower = map[Opcode][1];

    MachineBasicBlock* MBB = MI.getParent();
    DebugLoc dl = MI.getDebugLoc();
    MachineInstrBuilder Bu = BuildMI(*MBB, MI, dl, TI.get(OpcodeUpper));
    MachineInstrBuilder Bl = BuildMI(*MBB, MI, dl, TI.get(OpcodeLower));

    // VM512
    Bu.addReg(GetVM512Upper(MI.getOperand(0).getReg()));
    Bl.addReg(GetVM512Lower(MI.getOperand(0).getReg()));

    if (MI.getNumExplicitOperands() == 2) { // _Ml: VM512, VL
      // VL
      Bu.addReg(MI.getOperand(1).getReg());
      Bl.addReg(MI.getOperand(1).getReg());
    } else if (MI.getNumExplicitOperands() == 3) { // _Mvl: VM512, VR, VL
      // VR
      Bu.addReg(MI.getOperand(1).getReg());
      Bl.addReg(MI.getOperand(1).getReg());
      // VL
      Bu.addReg(MI.getOperand(2).getReg());
      Bl.addReg(MI.getOperand(2).getReg());
    } else if (MI.getNumExplicitOperands() == 4) { // _MvMl: VM512, VR, VM512, VL
      // VR
      Bu.addReg(MI.getOperand(1).getReg());
      Bl.addReg(MI.getOperand(1).getReg());
      // VM512
      Bu.addReg(GetVM512Upper(MI.getOperand(2).getReg()));
      Bl.addReg(GetVM512Lower(MI.getOperand(2).getReg()));
      // VL
      Bu.addReg(MI.getOperand(3).getReg());
      Bl.addReg(MI.getOperand(3).getReg());
    } else {
      report_fatal_error("unexpected number of operands for pvfmk");
    }

    MI.eraseFromParent();
}

bool VEInstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  case VE::EXTEND_STACK: {
    return expandExtendStackPseudo(MI);
  }
  case VE::EXTEND_STACK_GUARD: {
    MI.eraseFromParent(); // The pseudo instruction is gone now.
    return true;
  }
  case TargetOpcode::LOAD_STACK_GUARD: {
    assert(Subtarget.isTargetLinux() &&
           "Only Linux target is expected to contain LOAD_STACK_GUARD");
    report_fatal_error("expandPostRAPseudo for LOAD_STACK_GUARD is not implemented yet");
#if 0
    // offsetof(tcbhead_t, stack_guard) from sysdeps/sparc/nptl/tls.h in glibc.
    const int64_t Offset = Subtarget.is64Bit() ? 0x28 : 0x14;
    MI.setDesc(get(Subtarget.is64Bit() ? SP::LDXri : SP::LDri));
    MachineInstrBuilder(*MI.getParent()->getParent(), MI)
        .addReg(SP::G7)
        .addImm(Offset);
    return true;
#endif
  }
  case VE::GETSTACKTOP: {
    return expandGetStackTopPseudo(MI);
  }
#if 0
  case VE::VE_SELECT: {
    // (VESelect $dst, $CC, $condVal, $trueVal, $dst)
    //   -> (CMOVrr $dst, condCode, $trueVal, $condVal)
    // cmov.$df.$cf $dst, $trueval, $cond

    assert(MI.getOperand(0).getReg() == MI.getOperand(4).getReg());

    MachineBasicBlock* MBB = MI.getParent();
    DebugLoc dl = MI.getDebugLoc();
    BuildMI(*MBB, MI, dl, get(VE::CMOVWrr))
      .addReg(MI.getOperand(0).getReg())
      .addImm(MI.getOperand(1).getImm())
      .addReg(MI.getOperand(3).getReg())
      .addReg(MI.getOperand(2).getReg());

    MI.eraseFromParent();
    return true;
  }
#endif

  case VE::andm_MMM: buildVMRInst(MI, get(VE::andm_mmm)); return true;
  case VE::orm_MMM:  buildVMRInst(MI, get(VE::orm_mmm)); return true;
  case VE::xorm_MMM: buildVMRInst(MI, get(VE::xorm_mmm)); return true;
  case VE::eqvm_MMM: buildVMRInst(MI, get(VE::eqvm_mmm)); return true;
  case VE::nndm_MMM: buildVMRInst(MI, get(VE::nndm_mmm)); return true;
  case VE::negm_MM: buildVMRInst(MI, get(VE::negm_mm)); return true;

  case VE::LVMyir:
  case VE::LVMyim:
  case VE::LVMyir_y:
  case VE::LVMyim_y: {
    unsigned VMXu = GetVM512Upper(MI.getOperand(0).getReg());
    unsigned VMXl = GetVM512Lower(MI.getOperand(0).getReg());
    int64_t Imm = MI.getOperand(1).getImm();
    bool IsSrcReg =
        MI.getOpcode() == VE::LVMyir || MI.getOpcode() == VE::LVMyir_y;
    unsigned Src = IsSrcReg ? MI.getOperand(2).getReg() : VE::NoRegister;
    int64_t MImm = IsSrcReg ? 0 : MI.getOperand(2).getImm();
    bool KillSrc = IsSrcReg ? MI.getOperand(2).isKill() : false;
    unsigned VMX = VMXl;
    if (Imm >= 4) {
        VMX = VMXu;
        Imm -= 4;
    }
    MachineBasicBlock* MBB = MI.getParent();
    DebugLoc DL = MI.getDebugLoc();
    switch (MI.getOpcode()) {
    case VE::LVMyir:
      BuildMI(*MBB, MI, DL, get(VE::LVMxir))
        .addDef(VMX)
        .addImm(Imm)
        .addReg(Src, getKillRegState(KillSrc));
      break;
    case VE::LVMyim:
      BuildMI(*MBB, MI, DL, get(VE::LVMxim))
        .addDef(VMX)
        .addImm(Imm)
        .addImm(MImm);
      break;
    case VE::LVMyir_y:
      assert(MI.getOperand(0).getReg() == MI.getOperand(3).getReg() &&
             "LVMyir_y has different register in 3rd operand");
      BuildMI(*MBB, MI, DL, get(VE::LVMxir_x))
        .addDef(VMX)
        .addImm(Imm)
        .addReg(Src, getKillRegState(KillSrc))
        .addReg(VMX);
      break;
    case VE::LVMyim_y:
      assert(MI.getOperand(0).getReg() == MI.getOperand(3).getReg() &&
             "LVMyim_y has different register in 3rd operand");
      BuildMI(*MBB, MI, DL, get(VE::LVMxim_x))
        .addDef(VMX)
        .addImm(Imm)
        .addImm(MImm)
        .addReg(VMX);
      break;
    }
    MI.eraseFromParent();
    return true;
  }
  case VE::lvm_MMIs: {
    unsigned VMXu = GetVM512Upper(MI.getOperand(0).getReg());
    unsigned VMXl = GetVM512Lower(MI.getOperand(0).getReg());
    unsigned VMDu = GetVM512Upper(MI.getOperand(1).getReg());
    unsigned VMDl = GetVM512Upper(MI.getOperand(1).getReg());
    int64_t Imm = MI.getOperand(2).getImm();
    unsigned VMX = VMXl;
    unsigned VMD = VMDl;
    if (Imm >= 4) {
        VMX = VMXu;
        VMD = VMDu;
        Imm -= 4;
    }
    MachineBasicBlock* MBB = MI.getParent();
    DebugLoc DL = MI.getDebugLoc();
    BuildMI(*MBB, MI, DL, get(VE::LVMxir_x), VMX)
      .addImm(Imm)
      .addReg(MI.getOperand(3).getReg())
      .addReg(VMD);
    MI.eraseFromParent();
    return true;
  }
  case VE::SVMyi:
  case VE::svm_sMI: {
    unsigned Dest = MI.getOperand(0).getReg();
    unsigned VMZu = GetVM512Upper(MI.getOperand(1).getReg());
    unsigned VMZl = GetVM512Lower(MI.getOperand(1).getReg());
    bool KillSrc = MI.getOperand(1).isKill();
    int64_t Imm = MI.getOperand(2).getImm();
    unsigned VMZ = VMZl;
    if (Imm >= 4) {
        VMZ = VMZu;
        Imm -= 4;
    }
    MachineBasicBlock* MBB = MI.getParent();
    DebugLoc DL = MI.getDebugLoc();
    MachineInstrBuilder MIB = BuildMI(*MBB, MI, DL, get(VE::SVMxi), Dest)
      .addReg(VMZ)
      .addImm(Imm);
    MachineInstr *Inst = MIB.getInstr();
    MI.eraseFromParent();
    if (KillSrc) {
      const TargetRegisterInfo *TRI = &getRegisterInfo();
      Inst->addRegisterKilled(MI.getOperand(1).getReg(), TRI, true);
    }
    return true;
  }
  case VE::pvfmkat_Ml:
  case VE::pvfmkaf_Ml:
  case VE::pvfmkwgt_Mvl: case VE::pvfmkwgt_MvMl:
  case VE::pvfmkwlt_Mvl: case VE::pvfmkwlt_MvMl:
  case VE::pvfmkwne_Mvl: case VE::pvfmkwne_MvMl:
  case VE::pvfmkweq_Mvl: case VE::pvfmkweq_MvMl:
  case VE::pvfmkwge_Mvl: case VE::pvfmkwge_MvMl:
  case VE::pvfmkwle_Mvl: case VE::pvfmkwle_MvMl:
  case VE::pvfmkwnum_Mvl: case VE::pvfmkwnum_MvMl:
  case VE::pvfmkwnan_Mvl: case VE::pvfmkwnan_MvMl:
  case VE::pvfmkwgtnan_Mvl: case VE::pvfmkwgtnan_MvMl:
  case VE::pvfmkwltnan_Mvl: case VE::pvfmkwltnan_MvMl:
  case VE::pvfmkwnenan_Mvl: case VE::pvfmkwnenan_MvMl:
  case VE::pvfmkweqnan_Mvl: case VE::pvfmkweqnan_MvMl:
  case VE::pvfmkwgenan_Mvl: case VE::pvfmkwgenan_MvMl:
  case VE::pvfmkwlenan_Mvl: case VE::pvfmkwlenan_MvMl:
  case VE::pvfmksgt_Mvl: case VE::pvfmksgt_MvMl:
  case VE::pvfmkslt_Mvl: case VE::pvfmkslt_MvMl:
  case VE::pvfmksne_Mvl: case VE::pvfmksne_MvMl:
  case VE::pvfmkseq_Mvl: case VE::pvfmkseq_MvMl:
  case VE::pvfmksge_Mvl: case VE::pvfmksge_MvMl:
  case VE::pvfmksle_Mvl: case VE::pvfmksle_MvMl:
  case VE::pvfmksnum_Mvl: case VE::pvfmksnum_MvMl:
  case VE::pvfmksnan_Mvl: case VE::pvfmksnan_MvMl:
  case VE::pvfmksgtnan_Mvl: case VE::pvfmksgtnan_MvMl:
  case VE::pvfmksltnan_Mvl: case VE::pvfmksltnan_MvMl:
  case VE::pvfmksnenan_Mvl: case VE::pvfmksnenan_MvMl:
  case VE::pvfmkseqnan_Mvl: case VE::pvfmkseqnan_MvMl:
  case VE::pvfmksgenan_Mvl: case VE::pvfmksgenan_MvMl:
  case VE::pvfmkslenan_Mvl: case VE::pvfmkslenan_MvMl: {
    expandPseudoVFMK_VL(*this, MI);
    return true;
  }
  }
  return false;
}

bool VEInstrInfo::expandExtendStackPseudo(MachineInstr &MI) const {
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  const VESubtarget &STI = MF.getSubtarget<VESubtarget>();
  const VEInstrInfo &TII = *STI.getInstrInfo();
  DebugLoc dl = MBB.findDebugLoc(MI);

  // Create following instructions and multiple basic blocks.
  //
  // thisBB:
  //   brge.l.t %sp, %sl, sinkBB
  // syscallBB:
  //   ld      %s61, 0x18(, %tp)        // load param area
  //   or      %s62, 0, %s0             // spill the value of %s0
  //   lea     %s63, 0x13b              // syscall # of grow
  //   shm.l   %s63, 0x0(%s61)          // store syscall # at addr:0
  //   shm.l   %sl, 0x8(%s61)           // store old limit at addr:8
  //   shm.l   %sp, 0x10(%s61)          // store new limit at addr:16
  //   monc                             // call monitor
  //   or      %s0, 0, %s62             // restore the value of %s0
  // sinkBB:

  // Create new MBB
  MachineBasicBlock *BB = &MBB;
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineBasicBlock *syscallMBB = MF.CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = MF.CreateMachineBasicBlock(LLVM_BB);
  MachineFunction::iterator It = ++(BB->getIterator());
  MF.insert(It, syscallMBB);
  MF.insert(It, sinkMBB);

  // Transfer the remainder of BB and its successor edges to sinkMBB.
  sinkMBB->splice(sinkMBB->begin(), BB,
                  std::next(std::next(MachineBasicBlock::iterator(MI))),
                  BB->end());
  sinkMBB->transferSuccessorsAndUpdatePHIs(BB);

  // Next, add the true and fallthrough blocks as its successors.
  BB->addSuccessor(syscallMBB);
  BB->addSuccessor(sinkMBB);
  BuildMI(BB, dl, TII.get(VE::BRCFLrr_t))
      .addImm(VECC::CC_IGE)
      .addReg(VE::SX11) // %sp
      .addReg(VE::SX8)  // %sl
      .addMBB(sinkMBB);

  BB = syscallMBB;

  // Update machine-CFG edges
  BB->addSuccessor(sinkMBB);

  BuildMI(BB, dl, TII.get(VE::LDrii), VE::SX61)
      .addReg(VE::SX14)
      .addImm(0)
      .addImm(0x18);
  BuildMI(BB, dl, TII.get(VE::ORri), VE::SX62)
      .addReg(VE::SX0)
      .addImm(0);
  BuildMI(BB, dl, TII.get(VE::LEAzii), VE::SX63)
      .addImm(0)
      .addImm(0)
      .addImm(0x13b);
  BuildMI(BB, dl, TII.get(VE::SHMLri))
      .addReg(VE::SX61)
      .addImm(0)
      .addReg(VE::SX63);
  BuildMI(BB, dl, TII.get(VE::SHMLri))
      .addReg(VE::SX61)
      .addImm(8)
      .addReg(VE::SX8);
  BuildMI(BB, dl, TII.get(VE::SHMLri))
      .addReg(VE::SX61)
      .addImm(16)
      .addReg(VE::SX11);
  BuildMI(BB, dl, TII.get(VE::MONC));

  BuildMI(BB, dl, TII.get(VE::ORri), VE::SX0)
      .addReg(VE::SX62)
      .addImm(0);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}

bool VEInstrInfo::expandGetStackTopPseudo(MachineInstr &MI) const {
  MachineBasicBlock *MBB = MI.getParent();
  MachineFunction &MF = *MBB->getParent();
  const VESubtarget &STI = MF.getSubtarget<VESubtarget>();
  const VEInstrInfo &TII = *STI.getInstrInfo();
  DebugLoc DL = MBB->findDebugLoc(MI);

  // Create following instruction
  //
  //   dst = %sp + target specific frame + the size of parameter area

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const VEFrameLowering &TFL = *STI.getFrameLowering();

  // The VE ABI requires a reserved 176 bytes area at the top
  // of stack as described in VESubtarget.cpp.  So, we adjust it here.
  unsigned NumBytes = STI.getAdjustedFrameSize(0);

  // Also adds the size of parameter area.
  if (MFI.adjustsStack() && TFL.hasReservedCallFrame(MF))
    NumBytes += MFI.getMaxCallFrameSize();

  BuildMI(*MBB, MI, DL, TII.get(VE::LEArii))
      .addDef(MI.getOperand(0).getReg())
      .addReg(VE::SX11)
      .addImm(0)
      .addImm(NumBytes);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}
