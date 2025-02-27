; RUN: llc < %s -mtriple=ve | FileCheck %s

%struct.__jmp_buf_tag = type { [25 x i64], i64, [16 x i64] }

@buf = common global [1 x %struct.__jmp_buf_tag] zeroinitializer, align 8

; Function Attrs: noinline nounwind optnone
define signext i32 @t_setjmp() {
; CHECK-LABEL: t_setjmp:
; CHECK:       .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    st %s18, 48(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s19, 56(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s20, 64(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s21, 72(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s22, 80(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s23, 88(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s24, 96(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s25, 104(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s26, 112(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s27, 120(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s28, 128(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s29, 136(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s30, 144(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s31, 152(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s32, 160(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s33, 168(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    lea %s0, buf@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, buf@hi(, %s0)
; CHECK-NEXT:    st %s9, (, %s0)
; CHECK-NEXT:    st %s11, 16(, %s0)
; CHECK-NEXT:    lea %s1, .LBB{{[0-9]+}}_3@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, .LBB{{[0-9]+}}_3@hi(, %s1)
; CHECK-NEXT:    st %s1, 8(, %s0)
; CHECK-NEXT:    # EH_SJlJ_SETUP .LBB{{[0-9]+}}_3
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    lea %s0, 0
; CHECK-NEXT:    br.l.t .LBB{{[0-9]+}}_2
; CHECK-NEXT:  .LBB{{[0-9]+}}_3: # Block address taken
; CHECK-NEXT:    lea %s0, 1
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    ld %s33, 168(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s32, 160(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s31, 152(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s30, 144(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s29, 136(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s28, 128(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s27, 120(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s26, 112(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s25, 104(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s24, 96(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s23, 88(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s22, 80(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s21, 72(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s20, 64(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s19, 56(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s18, 48(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    or %s11, 0, %s9
  %1 = call i8* @llvm.frameaddress(i32 0)
  store i8* %1, i8** bitcast ([1 x %struct.__jmp_buf_tag]* @buf to i8**), align 8
  %2 = call i8* @llvm.stacksave()
  store i8* %2, i8** getelementptr inbounds (i8*, i8** bitcast ([1 x %struct.__jmp_buf_tag]* @buf to i8**), i64 2), align 8
  %3 = call i32 @llvm.eh.sjlj.setjmp(i8* bitcast ([1 x %struct.__jmp_buf_tag]* @buf to i8*))
  ret i32 %3
}

; Function Attrs: nounwind readnone
declare i8* @llvm.frameaddress(i32)

; Function Attrs: nounwind
declare i8* @llvm.stacksave()

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(i8*)

; Function Attrs: noinline nounwind optnone
define void @t_longjmp() {
; CHECK-LABEL: t_longjmp:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, buf@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, buf@hi(, %s0)
; CHECK-NEXT:    ld %s9, (, %s0)
; CHECK-NEXT:    ld %s1, 8(, %s0)
; CHECK-NEXT:    or %s10, 0, %s0
; CHECK-NEXT:    ld %s11, 16(, %s0)
; CHECK-NEXT:    b.l.t (, %s1)
  call void @llvm.eh.sjlj.longjmp(i8* bitcast ([1 x %struct.__jmp_buf_tag]* @buf to i8*))
  unreachable
                                                  ; No predecessors!
  ret void
}

; Function Attrs: noreturn nounwind
declare void @llvm.eh.sjlj.longjmp(i8*)

