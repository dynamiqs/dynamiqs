; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_select_fusion_1(ptr noalias align 256 dereferenceable(65536) %0, ptr noalias align 256 dereferenceable(65536) %1, ptr noalias align 256 dereferenceable(1) %2, ptr noalias align 256 dereferenceable(65536) %3) #0 {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %7 = getelementptr inbounds [1 x i8], ptr %2, i32 0, i32 0
  %8 = load i8, ptr %7, align 1, !invariant.load !3
  %9 = mul i32 %5, 128
  %10 = add i32 %9, %6
  %11 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %10
  %12 = load { double, double }, ptr %11, align 8, !invariant.load !3
  %13 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %10
  %14 = load { double, double }, ptr %13, align 8
  %15 = trunc i8 %8 to i1
  %16 = select i1 %15, { double, double } %12, { double, double } %14
  store { double, double } %16, ptr %13, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 32}
!2 = !{i32 0, i32 128}
!3 = !{}
