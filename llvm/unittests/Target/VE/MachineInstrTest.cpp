#include "VEInstrInfo.h"
#include "VESubtarget.h"
#include "VETargetMachine.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "gtest/gtest.h"

using namespace llvm;

TEST(VETest, VLIndex) {
  using namespace VE;

  // Return expected VL register index in each MI's operands.  Aurora VE has
  // multiple instruction formats for each instruction.  So, we defines
  // instructions hierarchically and tests a part of whole instructions.
  // This function returns -1 to N as expected index, or -2 as default.
  // We skip a test on an instructions that this functions returns -2.
  auto VLIndex = [](unsigned Opcode) {
    switch (Opcode) {
    default:
      break;
    case VLDNCrz:
      return -1;
    case VLDUNCrzl:
    case VLDLSXrzl_v:
    case VLDLZXNCirL:
    case VLD2DNCrrL_v:
    case VLDU2DNCrzL_v:
    case VLDL2DSXizL_v:
    case VLDL2DZXNCirl:
      return 3;
    case VSTOTrrv:
      return -1;
    case VSTUNCrzvl:
    case VSTLNCOTizvL:
    case VST2Dirvl:
      return 3;
    case VSTU2DNCrzvml:
    case VSTL2DNCOTrzvxl:
      return 4;
    case VGTNCsrzm_v:
      return -1;
    case VGTUNCvrzl:
    case VGTLSXvrzl_v:
    case VGTLZXNCsirL:
      return 4;
    case VGTNCsrrmL_v:
    case VGTUNCvrzxL:
    case VGTLSXsizxl_v:
      return 5;
    case VSCNCsrzvm:
      return -1;
    case VSCUNCvrzvl:
    case VSCLNCsirvL:
      return 4;
    case VSCOTsrrvmL:
    case VSCUNCOTvrzvxL:
    case VSCLsizvxl:
      return 5;
    case PFCHVrr:
      return -1;
    case PFCHVrrl:
    case PFCHVNCrzL:
      return 2;
    case VBRDrm:
      return -1;
    case VBRDrl:
      return 2;
    case VBRDixL_v:
      return 3;
    case VMVrvx_v:
      return -1;
    case VMVivl:
      return 3;
    case VMVrvmL_v:
      return 4;
    case VADDULvv_v:
    case PVADDULOrvx:
      return -1;
    case VADDUWvvl_v:
    case PVADDUUPrvL:
      return 3;
    case PVADDUvvxL_v:
    case VADDSWSXivml:
    case VADDSLivml:
      return 4;
    case VDIVULvv_v:
    case VDIVSWSXrvx:
      return -1;
    case VDIVUWvrl_v:
    case VDIVSWZXviL:
      return 3;
    case VDIVSLivxL_v:
    case VDIVSWSXivml:
      return 4;
    // We test casually if instructions are defined using a multiclass already
    // tested.
    case VSUBSLivml:
    case VMULSLivml:
    case VCMPSLivml:
    case VMAXSLivml:
      return 4;
    case VANDvv_v:
    case PVANDLOrvx:
      return -1;
    case PVANDvvl_v:
    case PVANDUPrvL:
      return 3;
    case VORvvxL_v:
    case PVORLOmvml:
    case VXORmvml:
    case VEQVmvml:
      return 4;
    case VLDZv:
      return -1;
    case VPCNTvL:
      return 2;
    case VBRVvxl:
      return 3;
    case VSEQ:
      return -1;
    case VSEQL:
      return 1;
    case VSEQxl:
      return 2;
    case VSLLvv_v:
    case PVSLLLOvrx:
      return -1;
    case PVSLLvvl_v:
    case PVSRLUPvrL:
      return 3;
    case VSLLvixL_v:
    case PVSRLLOvrml:
    case VSLALvixL_v:
    case VSRALvixL_v:
      return 4;
    case VSLDvvr_v:
    case VSLDvvix:
      return -1;
    case VSLDvvrl_v:
    case VSRDvviL:
      return 4;
    case VSLDvvixL_v:
    case VSRDvvrml:
      return 5;
    case VSFAvrr_v:
    case VSFAvrmm_v:
      return -1;
    case VSFAvirl_v:
    case VSFAvirL:
      return 4;
    case VSFAvimml:
    case VSFAvimmL_v:
      return 5;
    case VFADDDivml:
    case VFSUBDivml:
    case VFMULDivml:
    case VFDIVDivml:
    case VFCMPDivml:
    case VFMAXDivml:
      return 4;
    case VFSQRTDv_v:
    case VFSQRTSvx:
      return -1;
    case VFSQRTDvl_v:
    case VFSQRTDvL:
      return 2;
    case VFSQRTDvxL_v:
    case VFSQRTDvml:
    case VFSQRTDvmL:
      return 3;
    case VFMADDvvv_v:
    case PVFMADLOvrvx:
      return -1;
    case PVFMADvivl_v:
    case PVFMADUPvrvL:
      return 4;
    case VFMADSivvxL_v:
    case PVFMADLOvrvml:
    case VFMSBDivvxL_v:
    case VFNMADDivvxL_v:
    case VFNMSBDivvxL_v:
      return 5;
    case VRCPDvmL:
    case VRSQRTDvmL:
    case VRSQRTDNEXvmL:
      return 3;
    case VCVTWDSXv:
    case VCVTWDZXvx_v:
      return -1;
    case VCVTWSSXvl_v:
    case VCVTWSZXvL:
      return 3;
    case PVCVTWSLOvxL_v:
    case PVCVTWSUPvml:
    case PVCVTWSvmL:
    case VCVTLDvml:
      return 4;
    case VCVTDWvml:
    case VCVTDLvml:
    case VCVTSDvml:
    case VCVTDSvml:
    case VSUMWSXvml:
    case VSUMLvml:
    case VFSUMDvml:
    case VRMAXSWFSTSXvml:
    case VRMAXSLFSTvml:
    case VFRMAXDFSTvml:
    case VRANDvml:
    case VRORvml:
    case VRXORvml:
      return 3;
    case VFIADvr_v:
    case VFIASvi_v:
      return -1;
    case VFIADvrl_v:
    case VFIASviL_v:
    case VFISDviL_v:
    case VFIMDviL_v:
      return 3;
    case VFIAMDvvr_v:
    case VFIAMSvvi_v:
      return -1;
    case VFISMDvvrl_v:
    case VFISMSvviL_v:
    case VFIMADvviL_v:
    case VFIMSDvviL_v:
      return 4;
    case VMRGivml:
      return 4;
    case VCPvxl:
    case VEXvxl:
      return 3;
    case VSHFvvr:
    case VSHFvvr_v:
      return -1;
    case VSHFvvrl:
    case VSHFvvrL_v:
      return 4;
    case VFMKLv:
    case VFMKLxvx:
      return -1;
    case VFMKLvl:
    case VFMKLxvL:
      return 3;
    case VFMKLvml:
    case VFMKLxvxL:
      return 4;
    case VFMKLal:
    case VFMKLxnaL:
      return 1;
    case VFMKLaml:
    case VFMKLxnaxL:
    case VFMKWxnaxL:
    case VFMKDxnaxL:
      return 2;
    case TOVMm:
    case PCVMx:
    case LZVMm:
      return -1;
    case TOVMml:
    case PCVMmL:
    case LZVMxl:
      return 2;
    }
    return -2;
  };

  LLVMInitializeVETargetInfo();
  LLVMInitializeVETarget();
  LLVMInitializeVETargetMC();

  auto TT(Triple::normalize("ve-none-none-eabi"));
  std::string Error;
  const Target *T = TargetRegistry::lookupTarget(TT, Error);
  if (!T) {
    dbgs() << Error;
    return;
  }

  TargetOptions Options;
  auto TM = std::unique_ptr<LLVMTargetMachine>(
    static_cast<LLVMTargetMachine*>(
      T->createTargetMachine(TT, "ve", "", Options, None, None,
                             CodeGenOpt::Default)));
  VESubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                 std::string(TM->getTargetFeatureString()),
                 *static_cast<const VETargetMachine *>(TM.get()));
  const VEInstrInfo *TII = ST.getInstrInfo();
  auto MII = TM->getMCInstrInfo();

  for (unsigned i = 0; i < VE::INSTRUCTION_LIST_END; ++i) {
    // Skip -2 (default value)
    if (VLIndex(i) == -2)
      continue;

    const MCInstrDesc &Desc = TII->get(i);

    uint64_t Flags = Desc.TSFlags;
    ASSERT_EQ(VLIndex(i), GET_VLINDEX(Flags))
              << MII->getName(i)
              << ": mismatched expected VL register index in its argument\n";
  }
}
