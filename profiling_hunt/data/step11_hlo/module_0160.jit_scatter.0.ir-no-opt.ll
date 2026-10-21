; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_dynamic_update_slice_fusion(ptr noalias align 256 dereferenceable(1024) %0, ptr noalias align 16 dereferenceable(4) %1, ptr noalias align 256 dereferenceable(16) %2, ptr noalias align 16 dereferenceable(16) %3, ptr noalias align 256 dereferenceable(1) %4, ptr noalias align 256 dereferenceable(1024) %5) #0 {
  %7 = getelementptr inbounds [1 x i32], ptr %1, i32 0, i32 0
  %8 = load i32, ptr %7, align 4, !invariant.load !1
  %9 = call i32 @llvm.smin.i32(i32 %8, i32 63)
  %10 = call i32 @llvm.smax.i32(i32 %9, i32 0)
  %11 = getelementptr inbounds [1 x i8], ptr %4, i32 0, i32 0
  %12 = load i8, ptr %11, align 1, !invariant.load !1
  %13 = getelementptr inbounds [1 x { double, double }], ptr %3, i32 0, i32 0
  %14 = load { double, double }, ptr %13, align 8, !invariant.load !1
  %15 = getelementptr inbounds [1 x { double, double }], ptr %2, i32 0, i32 0
  %16 = load { double, double }, ptr %15, align 8, !invariant.load !1
  %17 = trunc i8 %12 to i1
  %18 = select i1 %17, { double, double } %14, { double, double } %16
  %19 = getelementptr inbounds [64 x { double, double }], ptr %0, i32 0, i32 %10
  store { double, double } %18, ptr %19, align 8
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
