; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_dynamic_slice_fusion(ptr noalias align 16 dereferenceable(1024) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(16) %2) #0 {
  %4 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  %5 = load i64, ptr %4, align 4, !invariant.load !1
  %6 = call i64 @llvm.smin.i64(i64 %5, i64 63)
  %7 = call i64 @llvm.smax.i64(i64 %6, i64 0)
  %8 = getelementptr inbounds [64 x { double, double }], ptr %0, i32 0, i64 %7
  %9 = load { double, double }, ptr %8, align 8, !invariant.load !1
  %10 = getelementptr inbounds [1 x { double, double }], ptr %2, i32 0, i32 0
  store { double, double } %9, ptr %10, align 8
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #1

attributes #0 = { "nvvm.reqntid"="1,1,1" }
attributes #1 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
