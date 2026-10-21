; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_compare_fusion(ptr noalias align 16 dereferenceable(32) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(2) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %5 = getelementptr inbounds [2 x { double, double }], ptr %0, i32 0, i32 %4
  %6 = load { double, double }, ptr %5, align 8, !invariant.load !2
  %7 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  %8 = load i64, ptr %7, align 4, !invariant.load !2
  %9 = sitofp i64 %8 to double
  %10 = extractvalue { double, double } %6, 0
  %11 = extractvalue { double, double } %6, 1
  %12 = fcmp une double %10, %9
  %13 = fcmp une double %11, 0.000000e+00
  %14 = or i1 %12, %13
  %15 = zext i1 %14 to i8
  %16 = getelementptr inbounds [2 x i8], ptr %2, i32 0, i32 %4
  store i8 %15, ptr %16, align 1
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="2,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 2}
!2 = !{}
