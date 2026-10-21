; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_dynamic_slice_fusion(ptr noalias align 16 dereferenceable(1024) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(1008) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %5 = sext i32 %4 to i64
  %6 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  %7 = load i64, ptr %6, align 4, !invariant.load !2
  %8 = call i64 @llvm.smin.i64(i64 %7, i64 1)
  %9 = call i64 @llvm.smax.i64(i64 %8, i64 0)
  %10 = add i64 %5, %9
  %11 = getelementptr inbounds [64 x { double, double }], ptr %0, i32 0, i64 %10
  %12 = load { double, double }, ptr %11, align 8, !invariant.load !2
  %13 = getelementptr inbounds [63 x { double, double }], ptr %2, i32 0, i64 %5
  store { double, double } %12, ptr %13, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #2

attributes #0 = { "nvvm.reqntid"="63,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 63}
!2 = !{}
