; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_dynamic_update_slice_fusion_1(ptr noalias align 256 dereferenceable(320) %0, ptr noalias align 256 dereferenceable(320) %1, ptr noalias align 256 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(8) %3, ptr noalias align 256 dereferenceable(8) %4, ptr noalias align 256 dereferenceable(320) %5) #0 {
  %7 = getelementptr inbounds [1 x i64], ptr %4, i32 0, i32 0
  %8 = load i64, ptr %7, align 4, !invariant.load !1
  %9 = getelementptr inbounds [1 x i64], ptr %3, i32 0, i32 0
  %10 = load i64, ptr %9, align 4, !invariant.load !1
  %11 = add i64 %8, %10
  %12 = icmp slt i64 %11, 0
  %13 = add i64 %11, 40
  %14 = select i1 %12, i64 %13, i64 %11
  %15 = trunc i64 %14 to i32
  %16 = sext i32 %15 to i64
  %17 = call i64 @llvm.smin.i64(i64 %16, i64 39)
  %18 = call i64 @llvm.smax.i64(i64 %17, i64 0)
  %19 = icmp sge i32 %15, 0
  %20 = icmp sle i32 %15, 39
  %21 = and i1 %19, %20
  %22 = getelementptr inbounds [1 x i64], ptr %2, i32 0, i32 0
  %23 = load i64, ptr %22, align 4, !invariant.load !1
  %24 = add i64 %23, %10
  %25 = icmp slt i64 %24, 0
  %26 = add i64 %24, 40
  %27 = select i1 %25, i64 %26, i64 %24
  %28 = call i64 @llvm.smin.i64(i64 %27, i64 39)
  %29 = call i64 @llvm.smax.i64(i64 %28, i64 0)
  %30 = getelementptr inbounds [40 x double], ptr %1, i32 0, i64 %29
  %31 = load double, ptr %30, align 8, !invariant.load !1
  %32 = getelementptr inbounds [40 x double], ptr %0, i32 0, i64 %18
  %33 = load double, ptr %32, align 8
  %34 = select i1 %21, double %31, double %33
  store double %34, ptr %32, align 8
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #1

attributes #0 = { "nvvm.reqntid"="1,1,1" }
attributes #1 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
