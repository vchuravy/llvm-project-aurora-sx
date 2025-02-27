; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s
; ModuleID = 'gen/tests/vrminslfst_vvl.c'
source_filename = "gen/tests/vrminslfst_vvl.c"
target datalayout = "e-m:e-i64:64-n32:64-S128-v64:64:64-v128:64:64-v256:64:64-v512:64:64-v1024:64:64-v2048:64:64-v4096:64:64-v8192:64:64-v16384:64:64"
target triple = "ve-unknown-linux-gnu"

; Function Attrs: nounwind
define dso_local void @vrminslfst_vvl(i64* %0, i64* %1, i32 signext %2) local_unnamed_addr #0 {
; CHECK: vrmins.l.fst %v0, %v0
  %4 = icmp sgt i32 %2, 0
  br i1 %4, label %6, label %5

5:                                                ; preds = %6, %3
  ret void

6:                                                ; preds = %3, %6
  %7 = phi i64* [ %17, %6 ], [ %0, %3 ]
  %8 = phi i64* [ %18, %6 ], [ %1, %3 ]
  %9 = phi i32 [ %19, %6 ], [ 0, %3 ]
  %10 = sub nsw i32 %2, %9
  %11 = icmp slt i32 %10, 256
  %12 = select i1 %11, i32 %10, i32 256
  %13 = bitcast i64* %8 to i8*
  %14 = tail call <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %13, i32 %12)
  %15 = tail call <256 x double> @llvm.ve.vl.vrminslfst.vvl(<256 x double> %14, i32 %12)
  %16 = bitcast i64* %7 to i8*
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %15, i64 8, i8* %16, i32 %12)
  %17 = getelementptr inbounds i64, i64* %7, i64 256
  %18 = getelementptr inbounds i64, i64* %8, i64 256
  %19 = add nuw nsw i32 %9, 256
  %20 = icmp slt i32 %19, %2
  br i1 %20, label %6, label %5
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vld.vssl(i64, i8*, i32) #1

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vrminslfst.vvl(<256 x double>, i32) #2

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vst.vssl(<256 x double>, i64, i8*, i32) #3

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="-vec" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind writeonly }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.0 (git@socsv218.svp.cl.nec.co.jp:ve-llvm/llvm-project.git ea1e45464a3c0492368cbabae9242628b03e399d)"}
