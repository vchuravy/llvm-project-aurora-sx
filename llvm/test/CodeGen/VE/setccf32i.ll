; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s
; RUN: llc < %s -mtriple=ve-unknown-unknown -enable-no-nans-fp-math | FileCheck %s -check-prefix=NONANS

define zeroext i1 @setccaf(float, float) {
; CHECK-LABEL: setccaf:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
;
; NONANS-LABEL: setccaf:
; NONANS:       # %bb.0:
; NONANS-NEXT:    or %s0, 0, (0)1
; NONANS-NEXT:    b.l.t (, %s10)
  %3 = fcmp false float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccat(float, float) {
; CHECK-LABEL: setccat:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s0, 1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
;
; NONANS-LABEL: setccat:
; NONANS:       # %bb.0:
; NONANS-NEXT:    or %s0, 1, (0)1
; NONANS-NEXT:    b.l.t (, %s10)
  %3 = fcmp true float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccoeq(float, float) {
; CHECK-LABEL: setccoeq:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.s.eq %s1, (63)0, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
;
; NONANS-LABEL: setccoeq:
; NONANS:       # %bb.0:
; NONANS-NEXT:    or %s1, 0, (0)1
; NONANS-NEXT:    cmov.s.eq %s1, (63)0, %s0
; NONANS-NEXT:    or %s0, 0, %s1
; NONANS-NEXT:    b.l.t (, %s10)
  %3 = fcmp oeq float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccone(float, float) {
; CHECK-LABEL: setccone:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s1, 1084227584
; CHECK-NEXT:    fcmp.s %s1, %s0, %s1
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.s.ne %s0, (63)0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
;
; NONANS-LABEL: setccone:
; NONANS:       # %bb.0:
; NONANS-NEXT:    lea.sl %s1, 1084227584
; NONANS-NEXT:    fcmp.s %s1, %s0, %s1
; NONANS-NEXT:    or %s0, 0, (0)1
; NONANS-NEXT:    cmov.s.ne %s0, (63)0, %s1
; NONANS-NEXT:    b.l.t (, %s10)
  %3 = fcmp one float %0, 5.0
  ret i1 %3
}

define zeroext i1 @setccogt(float, float) {
; CHECK-LABEL: setccogt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.s.gt %s1, (63)0, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
;
; NONANS-LABEL: setccogt:
; NONANS:       # %bb.0:
; NONANS-NEXT:    or %s1, 0, (0)1
; NONANS-NEXT:    cmov.s.gt %s1, (63)0, %s0
; NONANS-NEXT:    or %s0, 0, %s1
; NONANS-NEXT:    b.l.t (, %s10)
  %3 = fcmp ogt float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccoge(float, float) {
; CHECK-LABEL: setccoge:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.s.ge %s1, (63)0, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
;
; NONANS-LABEL: setccoge:
; NONANS:       # %bb.0:
; NONANS-NEXT:    or %s1, 0, (0)1
; NONANS-NEXT:    cmov.s.ge %s1, (63)0, %s0
; NONANS-NEXT:    or %s0, 0, %s1
; NONANS-NEXT:    b.l.t (, %s10)
  %3 = fcmp oge float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccolt(float, float) {
; CHECK-LABEL: setccolt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.s.lt %s1, (63)0, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
;
; NONANS-LABEL: setccolt:
; NONANS:       # %bb.0:
; NONANS-NEXT:    or %s1, 0, (0)1
; NONANS-NEXT:    cmov.s.lt %s1, (63)0, %s0
; NONANS-NEXT:    or %s0, 0, %s1
; NONANS-NEXT:    b.l.t (, %s10)
  %3 = fcmp olt float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccole(float, float) {
; CHECK-LABEL: setccole:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.s.le %s1, (63)0, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
;
; NONANS-LABEL: setccole:
; NONANS:       # %bb.0:
; NONANS-NEXT:    or %s1, 0, (0)1
; NONANS-NEXT:    cmov.s.le %s1, (63)0, %s0
; NONANS-NEXT:    or %s0, 0, %s1
; NONANS-NEXT:    b.l.t (, %s10)
  %3 = fcmp ole float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccord(float, float) {
; CHECK-LABEL: setccord:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fcmp.s %s1, %s0, %s0
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.s.num %s0, (63)0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
;
; NONANS-LABEL: setccord:
; NONANS:       # %bb.0:
; NONANS-NEXT:    fcmp.s %s1, %s0, %s0
; NONANS-NEXT:    or %s0, 0, (0)1
; NONANS-NEXT:    cmov.s.num %s0, (63)0, %s1
; NONANS-NEXT:    b.l.t (, %s10)
  %3 = fcmp ord float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccuno(float, float) {
; CHECK-LABEL: setccuno:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fcmp.s %s1, %s0, %s0
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.s.nan %s0, (63)0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
;
; NONANS-LABEL: setccuno:
; NONANS:       # %bb.0:
; NONANS-NEXT:    fcmp.s %s1, %s0, %s0
; NONANS-NEXT:    or %s0, 0, (0)1
; NONANS-NEXT:    cmov.s.nan %s0, (63)0, %s1
; NONANS-NEXT:    b.l.t (, %s10)
  %3 = fcmp uno float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccueq(float, float) {
; CHECK-LABEL: setccueq:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.s.eqnan %s1, (63)0, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
;
; NONANS-LABEL: setccueq:
; NONANS:       # %bb.0:
; NONANS-NEXT:    or %s1, 0, (0)1
; NONANS-NEXT:    cmov.s.eq %s1, (63)0, %s0
; NONANS-NEXT:    or %s0, 0, %s1
; NONANS-NEXT:    b.l.t (, %s10)
  %3 = fcmp ueq float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccune(float, float) {
; CHECK-LABEL: setccune:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.s.nenan %s1, (63)0, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
;
; NONANS-LABEL: setccune:
; NONANS:       # %bb.0:
; NONANS-NEXT:    or %s1, 0, (0)1
; NONANS-NEXT:    cmov.s.ne %s1, (63)0, %s0
; NONANS-NEXT:    or %s0, 0, %s1
; NONANS-NEXT:    b.l.t (, %s10)
  %3 = fcmp une float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccugt(float, float) {
; CHECK-LABEL: setccugt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.s.gtnan %s1, (63)0, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
;
; NONANS-LABEL: setccugt:
; NONANS:       # %bb.0:
; NONANS-NEXT:    or %s1, 0, (0)1
; NONANS-NEXT:    cmov.s.gt %s1, (63)0, %s0
; NONANS-NEXT:    or %s0, 0, %s1
; NONANS-NEXT:    b.l.t (, %s10)
  %3 = fcmp ugt float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccuge(float, float) {
; CHECK-LABEL: setccuge:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.s.genan %s1, (63)0, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
;
; NONANS-LABEL: setccuge:
; NONANS:       # %bb.0:
; NONANS-NEXT:    or %s1, 0, (0)1
; NONANS-NEXT:    cmov.s.ge %s1, (63)0, %s0
; NONANS-NEXT:    or %s0, 0, %s1
; NONANS-NEXT:    b.l.t (, %s10)
  %3 = fcmp uge float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccult(float, float) {
; CHECK-LABEL: setccult:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.s.ltnan %s1, (63)0, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
;
; NONANS-LABEL: setccult:
; NONANS:       # %bb.0:
; NONANS-NEXT:    or %s1, 0, (0)1
; NONANS-NEXT:    cmov.s.lt %s1, (63)0, %s0
; NONANS-NEXT:    or %s0, 0, %s1
; NONANS-NEXT:    b.l.t (, %s10)
  %3 = fcmp ult float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccule(float, float) {
; CHECK-LABEL: setccule:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.s.lenan %s1, (63)0, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
;
; NONANS-LABEL: setccule:
; NONANS:       # %bb.0:
; NONANS-NEXT:    or %s1, 0, (0)1
; NONANS-NEXT:    cmov.s.le %s1, (63)0, %s0
; NONANS-NEXT:    or %s0, 0, %s1
; NONANS-NEXT:    b.l.t (, %s10)
  %3 = fcmp ule float %0, 0.0
  ret i1 %3
}
