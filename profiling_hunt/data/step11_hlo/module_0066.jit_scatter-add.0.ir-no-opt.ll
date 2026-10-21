; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_complex_fusion(ptr noalias align 16 dereferenceable(4) %0, ptr noalias align 16 dereferenceable(1008) %1, ptr noalias align 16 dereferenceable(1024) %2, ptr noalias align 256 dereferenceable(1024) %3) #0 {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %6 = getelementptr inbounds [1 x i32], ptr %0, i32 0, i32 0
  %7 = load i32, ptr %6, align 4, !invariant.load !2
  %8 = call i32 @llvm.smin.i32(i32 %7, i32 1)
  %9 = call i32 @llvm.smax.i32(i32 %8, i32 0)
  %10 = add i32 %9, 63
  %11 = icmp sge i32 %5, %9
  %12 = icmp slt i32 %5, %10
  %13 = and i1 %11, %12
  %14 = sub i32 %5, %9
  br i1 %13, label %15, label %29

15:                                               ; preds = %4
  %16 = getelementptr inbounds [63 x { double, double }], ptr %1, i32 0, i32 %14
  %17 = load { double, double }, ptr %16, align 8, !invariant.load !2
  %18 = extractvalue { double, double } %17, 0
  %19 = icmp sge i32 %7, 0
  %20 = icmp sle i32 %7, 1
  %21 = and i1 %19, %20
  %22 = call double @fused_complex_scatter_add_19_10(ptr %0, ptr %1, ptr %2, i32 %5)
  %23 = fadd double %22, %18
  %24 = select i1 %21, double %23, double %22
  %25 = extractvalue { double, double } %17, 1
  %26 = call double @fused_complex_scatter_add_22_10(ptr %0, ptr %1, ptr %2, i32 %5)
  %27 = fadd double %26, %25
  %28 = select i1 %21, double %27, double %26
  br label %32

29:                                               ; preds = %4
  %30 = call double @fused_complex_scatter_add_19_10(ptr %0, ptr %1, ptr %2, i32 %5)
  %31 = call double @fused_complex_scatter_add_22_10(ptr %0, ptr %1, ptr %2, i32 %5)
  br label %32

32:                                               ; preds = %15, %29
  %33 = phi double [ %30, %29 ], [ %24, %15 ]
  %34 = phi double [ %31, %29 ], [ %28, %15 ]
  br label %35

35:                                               ; preds = %32
  %36 = insertvalue { double, double } poison, double %33, 0
  %37 = insertvalue { double, double } %36, double %34, 1
  %38 = getelementptr inbounds [64 x { double, double }], ptr %3, i32 0, i32 %5
  store { double, double } %37, ptr %38, align 8
  ret void
}

define internal double @fused_complex_scatter_add_22_10(ptr noalias %0, ptr noalias %1, ptr noalias %2, i32 %3) {
  %5 = getelementptr inbounds [64 x { double, double }], ptr %2, i32 0, i32 %3
  %6 = load { double, double }, ptr %5, align 8, !invariant.load !2
  %7 = extractvalue { double, double } %6, 1
  ret double %7
}

define internal double @fused_complex_scatter_add_19_10(ptr noalias %0, ptr noalias %1, ptr noalias %2, i32 %3) {
  %5 = getelementptr inbounds [64 x { double, double }], ptr %2, i32 0, i32 %3
  %6 = load { double, double }, ptr %5, align 8, !invariant.load !2
  %7 = extractvalue { double, double } %6, 0
  ret double %7
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

attributes #0 = { "nvvm.reqntid"="64,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 64}
!2 = !{}
