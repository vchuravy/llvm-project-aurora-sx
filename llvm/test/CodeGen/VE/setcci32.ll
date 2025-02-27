; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define zeroext i1 @setcceq(i32 signext, i32 signext) {
; CHECK-LABEL: setcceq:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.w.zx %s0, %s0, %s1
; CHECK-NEXT:    ldz %s0, %s0
; CHECK-NEXT:    srl %s0, %s0, 6
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp eq i32 %0, %1
  ret i1 %3
}

define zeroext i1 @setccne(i32 signext, i32 signext) {
; CHECK-LABEL: setccne:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.w.zx %s0, %s0, %s1
; CHECK-NEXT:    cmpu.w %s0, 0, %s0
; CHECK-NEXT:    srl %s0, %s0, 31
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp ne i32 %0, %1
  ret i1 %3
}

define zeroext i1 @setccugt(i32 signext, i32 signext) {
; CHECK-LABEL: setccugt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.w %s0, %s1, %s0
; CHECK-NEXT:    srl %s0, %s0, 31
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp ugt i32 %0, %1
  ret i1 %3
}

define zeroext i1 @setccuge(i32 signext, i32 signext) {
; CHECK-LABEL: setccuge:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.w %s0, %s0, %s1
; CHECK-NEXT:    srl %s0, %s0, 31
; CHECK-NEXT:    xor %s0, 1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp uge i32 %0, %1
  ret i1 %3
}

define zeroext i1 @setccult(i32 signext, i32 signext) {
; CHECK-LABEL: setccult:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.w %s0, %s0, %s1
; CHECK-NEXT:    srl %s0, %s0, 31
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp ult i32 %0, %1
  ret i1 %3
}

define zeroext i1 @setccule(i32 signext, i32 signext) {
; CHECK-LABEL: setccule:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.w %s0, %s1, %s0
; CHECK-NEXT:    srl %s0, %s0, 31
; CHECK-NEXT:    xor %s0, 1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp ule i32 %0, %1
  ret i1 %3
}

define zeroext i1 @setccsgt(i32 signext, i32 signext) {
; CHECK-LABEL: setccsgt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.w.zx %s0, %s1, %s0
; CHECK-NEXT:    srl %s0, %s0, 31
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp sgt i32 %0, %1
  ret i1 %3
}

define zeroext i1 @setccsge(i32 signext, i32 signext) {
; CHECK-LABEL: setccsge:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.w.zx %s0, %s0, %s1
; CHECK-NEXT:    srl %s0, %s0, 31
; CHECK-NEXT:    xor %s0, 1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp sge i32 %0, %1
  ret i1 %3
}

define zeroext i1 @setccslt(i32 signext, i32 signext) {
; CHECK-LABEL: setccslt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.w.zx %s0, %s0, %s1
; CHECK-NEXT:    srl %s0, %s0, 31
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp slt i32 %0, %1
  ret i1 %3
}

define zeroext i1 @setccsle(i32 signext, i32 signext) {
; CHECK-LABEL: setccsle:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.w.zx %s0, %s1, %s0
; CHECK-NEXT:    srl %s0, %s0, 31
; CHECK-NEXT:    xor %s0, 1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp sle i32 %0, %1
  ret i1 %3
}
