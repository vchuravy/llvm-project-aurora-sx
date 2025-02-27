; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define void @func0(i1 signext %p, i8* %a) {
; CHECK-LABEL: func0:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st1b %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %p.conv = sext i1 %p to i8
  store i8 %p.conv, i8* %a, align 2
  ret void
}

define void @func1(i8 signext %p, i16* %a) {
; CHECK-LABEL: func1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st2b %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %p.conv = sext i8 %p to i16
  store i16 %p.conv, i16* %a, align 2
  ret void
}

define void @func2(i8 signext %p, i32* %a) {
; CHECK-LABEL: func2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    stl %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %p.conv = sext i8 %p to i32
  store i32 %p.conv, i32* %a, align 4
  ret void
}

define void @func3(i8 signext %p, i64* %a) {
; CHECK-LABEL: func3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %p.conv = sext i8 %p to i64
  store i64 %p.conv, i64* %a, align 8
  ret void
}

define void @func4(i8 signext %p, i128* %a) {
; CHECK-LABEL: func4:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s2, %s0, 63
; CHECK-NEXT:    st %s0, (, %s1)
; CHECK-NEXT:    st %s2, 8(, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %p.conv = sext i8 %p to i128
  store i128 %p.conv, i128* %a, align 8
  ret void
}

define void @func5(i16 signext %p, i32* %a) {
; CHECK-LABEL: func5:
; CHECK:       # %bb.0:
; CHECK-NEXT:    stl %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %p.conv = sext i16 %p to i32
  store i32 %p.conv, i32* %a, align 4
  ret void
}

define void @func6(i16 signext %p, i64* %a) {
; CHECK-LABEL: func6:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %p.conv = sext i16 %p to i64
  store i64 %p.conv, i64* %a, align 8
  ret void
}

define void @func7(i16 signext %p, i128* %a) {
; CHECK-LABEL: func7:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s2, %s0, 63
; CHECK-NEXT:    st %s0, (, %s1)
; CHECK-NEXT:    st %s2, 8(, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %p.conv = sext i16 %p to i128
  store i128 %p.conv, i128* %a, align 8
  ret void
}

define void @func8(i32 signext %p, i64* %a) {
; CHECK-LABEL: func8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %p.conv = sext i32 %p to i64
  store i64 %p.conv, i64* %a, align 8
  ret void
}

define void @func9(i32 signext %p, i128* %a) {
; CHECK-LABEL: func9:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s2, %s0, 63
; CHECK-NEXT:    st %s0, (, %s1)
; CHECK-NEXT:    st %s2, 8(, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %p.conv = sext i32 %p to i128
  store i128 %p.conv, i128* %a, align 8
  ret void
}

define void @func10(i64 %p, i128* %a) {
; CHECK-LABEL: func10:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s2, %s0, 63
; CHECK-NEXT:    st %s0, (, %s1)
; CHECK-NEXT:    st %s2, 8(, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %p.conv = sext i64 %p to i128
  store i128 %p.conv, i128* %a, align 8
  ret void
}
