; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_dynamic_slice_fusion(ptr noalias align 256 dereferenceable(1024) %0, ptr noalias align 16 dereferenceable(4) %1, ptr noalias align 256 dereferenceable(16) %2) #0 {
  %4 = getelementptr inbounds [1 x i32], ptr %1, i32 0, i32 0
  %5 = load i32, ptr %4, align 4, !invariant.load !1
  %6 = call i32 @llvm.smin.i32(i32 %5, i32 63)
  %7 = call i32 @llvm.smax.i32(i32 %6, i32 0)
  %8 = getelementptr inbounds [64 x { double, double }], ptr %0, i32 0, i32 %7
  %9 = load { double, double }, ptr %8, align 8, !invariant.load !1
  %10 = getelementptr inbounds [1 x { double, double }], ptr %2, i32 0, i32 0
  store { double, double } %9, ptr %10, align 8
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #1

attributes #0 = { "nvvm.reqntid"="1,1,1" }
attributes #1 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
