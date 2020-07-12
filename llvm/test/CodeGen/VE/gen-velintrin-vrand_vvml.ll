; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s
; ModuleID = 'gen/tests/vrand_vvml.c'
source_filename = "gen/tests/vrand_vvml.c"
target datalayout = "e-m:e-i64:64-n32:64-S64-v64:64:64-v128:64:64-v256:64:64-v512:64:64-v1024:64:64-v2048:64:64-v4096:64:64-v8192:64:64-v16384:64:64"
target triple = "ve"

; Function Attrs: nounwind
define dso_local void @vrand_vvml(i64*, i64*, i32*, i32) local_unnamed_addr #0 {
; CHECK: vrand %v0,%v0,%vm1
  %5 = icmp sgt i32 %3, 0
  br i1 %5, label %7, label %6

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i64* [ %22, %7 ], [ %0, %4 ]
  %9 = phi i64* [ %23, %7 ], [ %1, %4 ]
  %10 = phi i32* [ %24, %7 ], [ %2, %4 ]
  %11 = phi i32 [ %25, %7 ], [ 0, %4 ]
  %12 = sub nsw i32 %3, %11
  %13 = icmp slt i32 %12, 256
  %14 = select i1 %13, i32 %12, i32 256
  %15 = bitcast i64* %9 to i8*
  %16 = tail call <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %15, i32 %14)
  %17 = bitcast i32* %10 to i8*
  %18 = tail call <256 x double> @llvm.ve.vl.vldlzx.vssl(i64 4, i8* %17, i32 %14)
  %19 = tail call <256 x i1> @llvm.ve.vl.vfmkwgt.mvl(<256 x double> %18, i32 %14)
  %20 = bitcast i64* %8 to i8*
  %21 = tail call <256 x double> @llvm.ve.vl.vrand.vvml(<256 x double> %16, <256 x i1> %19, i32 %14)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %21, i64 8, i8* %20, i32 %14)
  %22 = getelementptr inbounds i64, i64* %8, i64 256
  %23 = getelementptr inbounds i64, i64* %9, i64 256
  %24 = getelementptr inbounds i32, i32* %10, i64 256
  %25 = add nuw nsw i32 %11, 256
  %26 = icmp slt i32 %25, %3
  br i1 %26, label %7, label %6
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vld.vssl(i64, i8*, i32) #1

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldlzx.vssl(i64, i8*, i32) #1

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwgt.mvl(<256 x double>, i32) #2

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vrand.vvml(<256 x double>, <256 x i1>, i32) #2

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vst.vssl(<256 x double>, i64, i8*, i32) #3

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind writeonly }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 9.0.0 (git@socsv218.svp.cl.nec.co.jp:ve-llvm/clang.git 166ce7eaa48ef1c8891ad1012b2f5819d7674e19) (llvm/llvm.git 538e6ca3317a129b1e492a725935d84bb0a64c7f)"}
