; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_pad_fusion(ptr noalias align 16 dereferenceable(1008) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(1024) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %5 = icmp sge i32 %4, 1
  br i1 %5, label %6, label %10

6:                                                ; preds = %3
  %7 = add i32 %4, -1
  %8 = getelementptr inbounds [63 x { double, double }], ptr %0, i32 0, i32 %7
  %9 = load { double, double }, ptr %8, align 8, !invariant.load !2
  br label %16

10:                                               ; preds = %3
  %11 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  %12 = load i64, ptr %11, align 4, !invariant.load !2
  %13 = sitofp i64 %12 to double
  %14 = insertvalue { double, double } poison, double %13, 0
  %15 = insertvalue { double, double } %14, double 0.000000e+00, 1
  br label %16

16:                                               ; preds = %6, %10
  %17 = phi { double, double } [ %15, %10 ], [ %9, %6 ]
  br label %18

18:                                               ; preds = %16
  %19 = getelementptr inbounds [64 x { double, double }], ptr %2, i32 0, i32 %4
  store { double, double } %17, ptr %19, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="64,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 64}
!2 = !{}
