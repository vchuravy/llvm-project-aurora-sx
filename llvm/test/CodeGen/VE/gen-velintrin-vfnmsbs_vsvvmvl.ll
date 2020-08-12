; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s
; ModuleID = 'gen/tests/vfnmsbs_vsvvmvl.c'
source_filename = "gen/tests/vfnmsbs_vsvvmvl.c"
target datalayout = "e-m:e-i64:64-n32:64-S64-v64:64:64-v128:64:64-v256:64:64-v512:64:64-v1024:64:64-v2048:64:64-v4096:64:64-v8192:64:64-v16384:64:64"
target triple = "ve"

; Function Attrs: nounwind
define dso_local void @vfnmsbs_vsvvmvl(float*, float, float*, float*, i32*, float*, i32) local_unnamed_addr #0 {
; CHECK: vfnmsb.s %v3,%s1,%v0,%v1,%vm1
  %8 = icmp sgt i32 %6, 0
  br i1 %8, label %10, label %9

9:                                                ; preds = %10, %7
  ret void

10:                                               ; preds = %7, %10
  %11 = phi float* [ %31, %10 ], [ %0, %7 ]
  %12 = phi float* [ %32, %10 ], [ %2, %7 ]
  %13 = phi float* [ %33, %10 ], [ %3, %7 ]
  %14 = phi i32* [ %34, %10 ], [ %4, %7 ]
  %15 = phi float* [ %35, %10 ], [ %5, %7 ]
  %16 = phi i32 [ %36, %10 ], [ 0, %7 ]
  %17 = sub nsw i32 %6, %16
  %18 = icmp slt i32 %17, 256
  %19 = select i1 %18, i32 %17, i32 256
  %20 = bitcast float* %12 to i8*
  %21 = tail call <256 x double> @llvm.ve.vl.vldu.vssl(i64 4, i8* %20, i32 %19)
  %22 = bitcast float* %13 to i8*
  %23 = tail call <256 x double> @llvm.ve.vl.vldu.vssl(i64 4, i8* %22, i32 %19)
  %24 = bitcast i32* %14 to i8*
  %25 = tail call <256 x double> @llvm.ve.vl.vldlzx.vssl(i64 4, i8* %24, i32 %19)
  %26 = tail call <256 x i1> @llvm.ve.vl.vfmkwgt.mvl(<256 x double> %25, i32 %19)
  %27 = bitcast float* %15 to i8*
  %28 = tail call <256 x double> @llvm.ve.vl.vldu.vssl(i64 4, i8* %27, i32 %19)
  %29 = bitcast float* %11 to i8*
  %30 = tail call <256 x double> @llvm.ve.vl.vfnmsbs.vsvvmvl(float %1, <256 x double> %21, <256 x double> %23, <256 x i1> %26, <256 x double> %28, i32 %19)
  tail call void @llvm.ve.vl.vstu.vssl(<256 x double> %30, i64 4, i8* %29, i32 %19)
  %31 = getelementptr inbounds float, float* %11, i64 256
  %32 = getelementptr inbounds float, float* %12, i64 256
  %33 = getelementptr inbounds float, float* %13, i64 256
  %34 = getelementptr inbounds i32, i32* %14, i64 256
  %35 = getelementptr inbounds float, float* %15, i64 256
  %36 = add nuw nsw i32 %16, 256
  %37 = icmp slt i32 %36, %6
  br i1 %37, label %10, label %9
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldu.vssl(i64, i8*, i32) #1

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldlzx.vssl(i64, i8*, i32) #1

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwgt.mvl(<256 x double>, i32) #2

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbs.vsvvmvl(float, <256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32) #2

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstu.vssl(<256 x double>, i64, i8*, i32) #3

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind writeonly }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 9.0.0 (git@socsv218.svp.cl.nec.co.jp:ve-llvm/clang.git 166ce7eaa48ef1c8891ad1012b2f5819d7674e19) (llvm/llvm.git 538e6ca3317a129b1e492a725935d84bb0a64c7f)"}
