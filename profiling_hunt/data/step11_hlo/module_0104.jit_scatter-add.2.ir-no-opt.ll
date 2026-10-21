; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_complex_fusion(ptr noalias align 16 dereferenceable(4) %0, ptr noalias align 16 dereferenceable(2048) %1, ptr noalias align 16 dereferenceable(1024) %2, ptr noalias align 256 dereferenceable(1) %3, ptr noalias align 256 dereferenceable(2048) %4) #0 {
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %7 = udiv i32 %6, 64
  %8 = urem i32 %6, 64
  %9 = getelementptr inbounds [1 x i32], ptr %0, i32 0, i32 0
  %10 = load i32, ptr %9, align 4, !invariant.load !2
  %11 = call i32 @llvm.smin.i32(i32 %10, i32 1)
  %12 = call i32 @llvm.smax.i32(i32 %11, i32 0)
  %13 = add i32 %12, 1
  %14 = icmp sge i32 %7, %12
  %15 = icmp slt i32 %7, %13
  %16 = and i1 %14, %15
  br i1 %16, label %17, label %31

17:                                               ; preds = %5
  %18 = getelementptr inbounds [64 x { double, double }], ptr %2, i32 0, i32 %8
  %19 = load { double, double }, ptr %18, align 8, !invariant.load !2
  %20 = extractvalue { double, double } %19, 0
  %21 = getelementptr inbounds [1 x i8], ptr %3, i32 0, i32 0
  %22 = load i8, ptr %21, align 1, !invariant.load !2
  %23 = call double @fused_complex_scatter_add_19_4(ptr %0, ptr %1, ptr %2, ptr %3, i32 %7, i32 %8)
  %24 = fadd double %23, %20
  %25 = trunc i8 %22 to i1
  %26 = select i1 %25, double %24, double %23
  %27 = extractvalue { double, double } %19, 1
  %28 = call double @fused_complex_scatter_add_22_4(ptr %0, ptr %1, ptr %2, ptr %3, i32 %7, i32 %8)
  %29 = fadd double %28, %27
  %30 = select i1 %25, double %29, double %28
  br label %34

31:                                               ; preds = %5
  %32 = call double @fused_complex_scatter_add_19_4(ptr %0, ptr %1, ptr %2, ptr %3, i32 %7, i32 %8)
  %33 = call double @fused_complex_scatter_add_22_4(ptr %0, ptr %1, ptr %2, ptr %3, i32 %7, i32 %8)
  br label %34

34:                                               ; preds = %17, %31
  %35 = phi double [ %32, %31 ], [ %26, %17 ]
  %36 = phi double [ %33, %31 ], [ %30, %17 ]
  br label %37

37:                                               ; preds = %34
  %38 = insertvalue { double, double } poison, double %35, 0
  %39 = insertvalue { double, double } %38, double %36, 1
  %40 = getelementptr inbounds [128 x { double, double }], ptr %4, i32 0, i32 %6
  store { double, double } %39, ptr %40, align 8
  ret void
}

define internal double @fused_complex_scatter_add_22_4(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, i32 %4, i32 %5) {
  %7 = mul i32 %4, 64
  %8 = add i32 %7, %5
  %9 = getelementptr inbounds [128 x { double, double }], ptr %1, i32 0, i32 %8
  %10 = load { double, double }, ptr %9, align 8, !invariant.load !2
  %11 = extractvalue { double, double } %10, 1
  ret double %11
}

define internal double @fused_complex_scatter_add_19_4(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, i32 %4, i32 %5) {
  %7 = mul i32 %4, 64
  %8 = add i32 %7, %5
  %9 = getelementptr inbounds [128 x { double, double }], ptr %1, i32 0, i32 %8
  %10 = load { double, double }, ptr %9, align 8, !invariant.load !2
  %11 = extractvalue { double, double } %10, 0
  ret double %11
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 128}
!2 = !{}
