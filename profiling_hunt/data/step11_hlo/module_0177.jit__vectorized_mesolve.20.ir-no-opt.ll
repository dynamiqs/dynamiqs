; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_select_fusion(ptr noalias align 256 dereferenceable(65536) %0, ptr noalias align 256 dereferenceable(1) %1, ptr noalias align 256 dereferenceable(65536) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = getelementptr inbounds [1 x i8], ptr %1, i32 0, i32 0
  %7 = load i8, ptr %6, align 1, !invariant.load !3
  %8 = mul i32 %4, 128
  %9 = add i32 %8, %5
  %10 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %9
  %11 = load { double, double }, ptr %10, align 8, !invariant.load !3
  %12 = trunc i8 %7 to i1
  %13 = select i1 %12, { double, double } zeroinitializer, { double, double } %11
  %14 = getelementptr inbounds [4096 x { double, double }], ptr %2, i32 0, i32 %9
  store { double, double } %13, ptr %14, align 8
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
