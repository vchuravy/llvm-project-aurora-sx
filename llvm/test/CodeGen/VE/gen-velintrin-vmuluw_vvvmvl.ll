; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s
; ModuleID = 'gen/tests/vmuluw_vvvmvl.c'
source_filename = "gen/tests/vmuluw_vvvmvl.c"
target datalayout = "e-m:e-i64:64-n32:64-S128-v64:64:64-v128:64:64-v256:64:64-v512:64:64-v1024:64:64-v2048:64:64-v4096:64:64-v8192:64:64-v16384:64:64"
target triple = "ve-unknown-linux-gnu"

; Function Attrs: nounwind
define dso_local void @vmuluw_vvvmvl(i32* %0, i32* %1, i32* %2, i32* %3, i32* %4, i32 signext %5) local_unnamed_addr #0 {
; CHECK: vmulu.w %v3, %v0, %v1, %vm1
  %7 = icmp sgt i32 %5, 0
  br i1 %7, label %9, label %8

8:                                                ; preds = %9, %6
  ret void

9:                                                ; preds = %6, %9
  %10 = phi i32* [ %30, %9 ], [ %0, %6 ]
  %11 = phi i32* [ %31, %9 ], [ %1, %6 ]
  %12 = phi i32* [ %32, %9 ], [ %2, %6 ]
  %13 = phi i32* [ %33, %9 ], [ %3, %6 ]
  %14 = phi i32* [ %34, %9 ], [ %4, %6 ]
  %15 = phi i32 [ %35, %9 ], [ 0, %6 ]
  %16 = sub nsw i32 %5, %15
  %17 = icmp slt i32 %16, 256
  %18 = select i1 %17, i32 %16, i32 256
  %19 = bitcast i32* %11 to i8*
  %20 = tail call <256 x double> @llvm.ve.vl.vldlsx.vssl(i64 4, i8* %19, i32 %18)
  %21 = bitcast i32* %12 to i8*
  %22 = tail call <256 x double> @llvm.ve.vl.vldlsx.vssl(i64 4, i8* %21, i32 %18)
  %23 = bitcast i32* %13 to i8*
  %24 = tail call <256 x double> @llvm.ve.vl.vldlzx.vssl(i64 4, i8* %23, i32 %18)
  %25 = tail call <256 x i1> @llvm.ve.vl.vfmkwgt.mvl(<256 x double> %24, i32 %18)
  %26 = bitcast i32* %14 to i8*
  %27 = tail call <256 x double> @llvm.ve.vl.vldlsx.vssl(i64 4, i8* %26, i32 %18)
  %28 = bitcast i32* %10 to i8*
  %29 = tail call <256 x double> @llvm.ve.vl.vmuluw.vvvmvl(<256 x double> %20, <256 x double> %22, <256 x i1> %25, <256 x double> %27, i32 %18)
  tail call void @llvm.ve.vl.vstl.vssl(<256 x double> %29, i64 4, i8* %28, i32 %18)
  %30 = getelementptr inbounds i32, i32* %10, i64 256
  %31 = getelementptr inbounds i32, i32* %11, i64 256
  %32 = getelementptr inbounds i32, i32* %12, i64 256
  %33 = getelementptr inbounds i32, i32* %13, i64 256
  %34 = getelementptr inbounds i32, i32* %14, i64 256
  %35 = add nuw nsw i32 %15, 256
  %36 = icmp slt i32 %35, %5
  br i1 %36, label %9, label %8
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldlsx.vssl(i64, i8*, i32) #1

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldlzx.vssl(i64, i8*, i32) #1

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwgt.mvl(<256 x double>, i32) #2

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vmuluw.vvvmvl(<256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32) #2

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstl.vssl(<256 x double>, i64, i8*, i32) #3

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="-vec" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind writeonly }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.0 (git@socsv218.svp.cl.nec.co.jp:ve-llvm/llvm-project.git ea1e45464a3c0492368cbabae9242628b03e399d)"}
