//===-- VEMCInstLower.cpp - Convert VE MachineInstr to MCInst -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower VE MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/VEMCExpr.h"
#include "VE.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"

using namespace llvm;

static MCOperand LowerSymbolOperand(const MachineInstr *MI,
                                    const MachineOperand &MO,
                                    const MCSymbol *Symbol, AsmPrinter &AP) {
  VEMCExpr::VariantKind Kind = (VEMCExpr::VariantKind)MO.getTargetFlags();

  const MCExpr *Expr = MCSymbolRefExpr::create(Symbol, AP.OutContext);
  // Add offset iff MO is not jump table info or machine basic block.
  if (!MO.isJTI() && !MO.isMBB() && MO.getOffset())
    Expr = MCBinaryExpr::createAdd(
        Expr, MCConstantExpr::create(MO.getOffset(), AP.OutContext),
        AP.OutContext);
  Expr = VEMCExpr::create(Kind, Expr, AP.OutContext);
  return MCOperand::createExpr(Expr);
}

static MCOperand LowerOperand(const MachineInstr *MI, const MachineOperand &MO,
                              AsmPrinter &AP) {
  switch (MO.getType()) {
  default:
    report_fatal_error("unsupported operand type");
    break;
  case MachineOperand::MO_CImmediate:
    report_fatal_error("unsupported MO_CImmediate operand type");
    break;
  case MachineOperand::MO_FPImmediate:
    report_fatal_error("unsupported MO_FPImmediate operand type");
    break;
  case MachineOperand::MO_FrameIndex:
    report_fatal_error("unsupported MO_FrameIndex operand type");
    break;
  case MachineOperand::MO_TargetIndex:
    report_fatal_error("unsupported MO_TargetIndex operand type");
    break;
  case MachineOperand::MO_JumpTableIndex:
    return LowerSymbolOperand(MI, MO, AP.GetJTISymbol(MO.getIndex()), AP);
  case MachineOperand::MO_RegisterLiveOut:
    report_fatal_error("unsupported MO_RegistrLiveOut operand type");
    break;
  case MachineOperand::MO_Metadata:
    report_fatal_error("unsupported MO_Metadata operand type");
    break;
  case MachineOperand::MO_MCSymbol:
    return LowerSymbolOperand(MI, MO, MO.getMCSymbol(), AP);
    break;
  case MachineOperand::MO_CFIIndex:
    report_fatal_error("unsupported MO_CFIIndex operand type");
    break;
  case MachineOperand::MO_IntrinsicID:
    report_fatal_error("unsupported MO_IntrinsicID operand type");
    break;
  case MachineOperand::MO_Predicate:
    report_fatal_error("unsupported MO_Predicate operand type");
    break;

  case MachineOperand::MO_Register:
    if (MO.isImplicit())
      break;
    return MCOperand::createReg(MO.getReg());

  case MachineOperand::MO_BlockAddress:
    return LowerSymbolOperand(
        MI, MO, AP.GetBlockAddressSymbol(MO.getBlockAddress()), AP);
  case MachineOperand::MO_ConstantPoolIndex:
    return LowerSymbolOperand(MI, MO, AP.GetCPISymbol(MO.getIndex()), AP);
  case MachineOperand::MO_ExternalSymbol:
    return LowerSymbolOperand(
        MI, MO, AP.GetExternalSymbolSymbol(MO.getSymbolName()), AP);
  case MachineOperand::MO_GlobalAddress:
    return LowerSymbolOperand(MI, MO, AP.getSymbol(MO.getGlobal()), AP);
  case MachineOperand::MO_Immediate:
    return MCOperand::createImm(MO.getImm());
  case MachineOperand::MO_MachineBasicBlock:
    return LowerSymbolOperand(MI, MO, MO.getMBB()->getSymbol(), AP);

  case MachineOperand::MO_RegisterMask:
    break;
  }
  return MCOperand();
}

void llvm::LowerVEMachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                       AsmPrinter &AP) {
  OutMI.setOpcode(MI->getOpcode());

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    MCOperand MCOp = LowerOperand(MI, MO, AP);

    if (MCOp.isValid())
      OutMI.addOperand(MCOp);
  }
}
