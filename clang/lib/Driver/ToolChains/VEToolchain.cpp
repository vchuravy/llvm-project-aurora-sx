//===--- VE.cpp - VE ToolChain Implementations ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VEToolchain.h"
#include "CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <cstdlib> // ::getenv

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

/// VE tool chain
VEToolChain::VEToolChain(const Driver &D, const llvm::Triple &Triple,
                         const ArgList &Args)
    : Linux(D, Triple, Args) {
  getProgramPaths().push_back("/opt/nec/ve/bin");
  // ProgramPaths are found via 'PATH' environment variable.

  // default file paths are:
  //   ${RESOURCEDIR}/lib/linux/ve (== getArchSpecificLibPath)
  //   /lib/../lib64
  //   /usr/lib/../lib64
  //   ${BINPATH}/../lib
  //   /lib
  //   /usr/lib
  //
  // These are OK for host, but no go for VE.  So, defines them all
  // from scratch here.
  getFilePaths().clear();
  getFilePaths().push_back(getArchSpecificLibPath());
  if (getTriple().isMusl())
    getFilePaths().push_back(computeSysRoot() + "/opt/nec/ve/musl/lib");
  else
    getFilePaths().push_back(computeSysRoot() + "/opt/nec/ve/lib");
}

Tool *VEToolChain::buildAssembler() const {
  return new tools::gnutools::Assembler(*this);
}

Tool *VEToolChain::buildLinker() const {
  return new tools::gnutools::Linker(*this);
}

bool VEToolChain::isPICDefault() const { return false; }

bool VEToolChain::isPIEDefault() const { return false; }

bool VEToolChain::isPICDefaultForced() const { return false; }

bool VEToolChain::SupportsProfiling() const { return false; }

bool VEToolChain::hasBlocksRuntime() const { return false; }

void VEToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                            ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(clang::driver::options::OPT_nostdinc))
    return;

  if (DriverArgs.hasArg(options::OPT_nobuiltininc) &&
      DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<128> P(getDriver().ResourceDir);
    llvm::sys::path::append(P, "include");
    addSystemInclude(DriverArgs, CC1Args, P);
  }

  if (!DriverArgs.hasArg(options::OPT_nostdlibinc)) {
    if (const char *cl_include_dir = getenv("NCC_C_INCLUDE_PATH")) {
      SmallVector<StringRef, 4> Dirs;
      const char EnvPathSeparatorStr[] = {llvm::sys::EnvPathSeparator, '\0'};
      StringRef(cl_include_dir).split(Dirs, StringRef(EnvPathSeparatorStr));
      ArrayRef<StringRef> DirVec(Dirs);
      addSystemIncludes(DriverArgs, CC1Args, DirVec);
    } else {
      if (getTriple().isMusl())
        addSystemInclude(DriverArgs, CC1Args,
                         getDriver().SysRoot + "/opt/nec/ve/musl/include");
      else
        addSystemInclude(DriverArgs, CC1Args,
                         getDriver().SysRoot + "/opt/nec/ve/include");
    }
  }
}

void VEToolChain::addClangTargetOptions(const ArgList &DriverArgs,
                                        ArgStringList &CC1Args,
                                        Action::OffloadKind) const {
  CC1Args.push_back("-nostdsysteminc");
  bool UseInitArrayDefault = true;
  if (!DriverArgs.hasFlag(options::OPT_fuse_init_array,
                          options::OPT_fno_use_init_array, UseInitArrayDefault))
    CC1Args.push_back("-fno-use-init-array");
}

void VEToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                               ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(clang::driver::options::OPT_nostdinc) ||
      DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;
  if (const char *cl_include_dir = getenv("NCC_CPLUS_INCLUDE_PATH")) {
    SmallVector<StringRef, 4> Dirs;
    const char EnvPathSeparatorStr[] = {llvm::sys::EnvPathSeparator, '\0'};
    StringRef(cl_include_dir).split(Dirs, StringRef(EnvPathSeparatorStr));
    ArrayRef<StringRef> DirVec(Dirs);
    addSystemIncludes(DriverArgs, CC1Args, DirVec);
  } else {
    SmallString<128> P(getDriver().ResourceDir);
    llvm::sys::path::append(P, "include/c++/v1");
    addSystemInclude(DriverArgs, CC1Args, P);
  }
}

void VEToolChain::AddCXXStdlibLibArgs(const ArgList &Args,
                                      ArgStringList &CmdArgs) const {
  assert((GetCXXStdlibType(Args) == ToolChain::CST_Libcxx) &&
         "Only -lc++ (aka libxx) is supported in this toolchain.");

  tools::addArchSpecificRPath(*this, Args, CmdArgs);

  CmdArgs.push_back("-lc++");
  CmdArgs.push_back("-lc++abi");
  CmdArgs.push_back("-lunwind");
  // libc++ requires -lpthread under glibc environment
  // libunwind requires -ldl under glibc environment
  if (!getTriple().isMusl()) {
    CmdArgs.push_back("-lpthread");
    CmdArgs.push_back("-ldl");
  }
}

llvm::ExceptionHandling
VEToolChain::GetExceptionModel(const ArgList &Args) const {
  // VE uses SjLj exceptions.
  return llvm::ExceptionHandling::SjLj;
}
