; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_dynamic_slice_fusion(ptr noalias align 256 dereferenceable(1024) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(1008) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %5 = getelementptr inbounds [2 x i32], ptr %1, i32 0, i32 1
  %6 = load i32, ptr %5, align 4, !invariant.load !2
  %7 = call i32 @llvm.smin.i32(i32 %6, i32 1)
  %8 = call i32 @llvm.smax.i32(i32 %7, i32 0)
  %9 = add i32 %4, %8
  %10 = getelementptr inbounds [64 x { double, double }], ptr %0, i32 0, i32 %9
  %11 = load { double, double }, ptr %10, align 8, !invariant.load !2
  %12 = getelementptr inbounds [63 x { double, double }], ptr %2, i32 0, i32 %4
  store { double, double } %11, ptr %12, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

attributes #0 = { "nvvm.reqntid"="63,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 63}
!2 = !{}
