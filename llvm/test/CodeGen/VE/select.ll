; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define fp128 @selectf128(i1 zeroext, fp128, fp128) {
; CHECK-LABEL: selectf128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.ne %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select i1 %0, fp128 %1, fp128 %2
  ret fp128 %4
}

define double @selectf64(i1 zeroext, double, double) {
; CHECK-LABEL: selectf64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s2, %s1, %s0
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select i1 %0, double %1, double %2
  ret double %4
}

define float @selectf32(i1 zeroext, float, float) {
; CHECK-LABEL: selectf32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s2, %s1, %s0
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select i1 %0, float %1, float %2
  ret float %4
}

define i64 @selecti64(i1 zeroext, i64, i64) {
; CHECK-LABEL: selecti64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s2, %s1, %s0
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select i1 %0, i64 %1, i64 %2
  ret i64 %4
}

define signext i32 @selecti32(i1 zeroext, i32 signext, i32 signext) {
; CHECK-LABEL: selecti32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s2, %s1, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s2, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select i1 %0, i32 %1, i32 %2
  ret i32 %4
}

define zeroext i1 @selecti1(i1 zeroext, i1 zeroext, i1 zeroext) {
; CHECK-LABEL: selecti1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s2, %s1, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select i1 %0, i1 %1, i1 %2
  ret i1 %4
}
