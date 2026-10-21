; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_dynamic_update_slice_fusion(ptr noalias align 256 dereferenceable(1024) %0, ptr noalias align 256 dereferenceable(1008) %1, ptr noalias align 16 dereferenceable(1008) %2, ptr noalias align 256 dereferenceable(1) %3, ptr noalias align 16 dereferenceable(8) %4, ptr noalias align 256 dereferenceable(1024) %5) #0 {
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %8 = getelementptr inbounds [2 x i32], ptr %4, i32 0, i32 1
  %9 = load i32, ptr %8, align 4, !invariant.load !2
  %10 = call i32 @llvm.smin.i32(i32 %9, i32 1)
  %11 = call i32 @llvm.smax.i32(i32 %10, i32 0)
  %12 = add i32 %7, %11
  %13 = getelementptr inbounds [1 x i8], ptr %3, i32 0, i32 0
  %14 = load i8, ptr %13, align 1, !invariant.load !2
  %15 = getelementptr inbounds [63 x { double, double }], ptr %2, i32 0, i32 %7
  %16 = load { double, double }, ptr %15, align 8, !invariant.load !2
  %17 = getelementptr inbounds [63 x { double, double }], ptr %1, i32 0, i32 %7
  %18 = load { double, double }, ptr %17, align 8, !invariant.load !2
  %19 = trunc i8 %14 to i1
  %20 = select i1 %19, { double, double } %16, { double, double } %18
  %21 = getelementptr inbounds [64 x { double, double }], ptr %0, i32 0, i32 %12
  store { double, double } %20, ptr %21, align 8
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
