; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_dynamic_update_slice_fusion(ptr noalias align 256 dereferenceable(458752) %0, ptr noalias align 256 dereferenceable(65536) %1, ptr noalias align 256 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(8) %3, ptr noalias align 256 dereferenceable(8) %4, ptr noalias align 256 dereferenceable(12) %5, ptr noalias align 256 dereferenceable(458752) %6) #0 {
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %9 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %10 = getelementptr inbounds [1 x i64], ptr %3, i32 0, i32 0
  %11 = load i64, ptr %10, align 4, !invariant.load !3
  %12 = getelementptr inbounds [1 x i64], ptr %4, i32 0, i32 0
  %13 = load i64, ptr %12, align 4, !invariant.load !3
  %14 = add i64 %11, %13
  %15 = icmp slt i64 %14, 0
  %16 = add i64 %14, 7
  %17 = select i1 %15, i64 %16, i64 %14
  %18 = trunc i64 %17 to i32
  %19 = call i32 @llvm.smin.i32(i32 %18, i32 6)
  %20 = call i32 @llvm.smax.i32(i32 %19, i32 0)
  %21 = getelementptr inbounds [3 x i32], ptr %5, i32 0, i32 0
  %22 = load i32, ptr %21, align 4, !invariant.load !3
  %23 = icmp sge i32 %18, 0
  %24 = icmp sge i32 %22, %18
  %25 = and i1 %23, %24
  %26 = zext i1 %25 to i8
  %27 = and i8 %26, 1
  %28 = getelementptr inbounds [3 x i32], ptr %5, i32 0, i32 1
  %29 = load i32, ptr %28, align 4, !invariant.load !3
  %30 = icmp sge i32 %29, 0
  %31 = zext i1 %30 to i8
  %32 = and i8 %27, %31
  %33 = getelementptr inbounds [3 x i32], ptr %5, i32 0, i32 2
  %34 = load i32, ptr %33, align 4, !invariant.load !3
  %35 = icmp sge i32 %34, 0
  %36 = zext i1 %35 to i8
  %37 = and i8 %32, %36
  %38 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  %39 = load double, ptr %38, align 8, !invariant.load !3
  %40 = mul i32 %8, 128
  %41 = add i32 %40, %9
  %42 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %41
  %43 = load { double, double }, ptr %42, align 8, !invariant.load !3
  %44 = extractvalue { double, double } %43, 0
  %45 = extractvalue { double, double } %43, 1
  %46 = fmul double %39, %44
  %47 = fmul double %45, 0.000000e+00
  %48 = fmul double %44, 0.000000e+00
  %49 = fmul double %39, %45
  %50 = fsub double %46, %47
  %51 = fadd double %48, %49
  %52 = insertvalue { double, double } poison, double %50, 0
  %53 = insertvalue { double, double } %52, double %51, 1
  %54 = mul i32 %20, 4096
  %55 = add i32 %54, %40
  %56 = add i32 %55, %9
  %57 = getelementptr inbounds [28672 x { double, double }], ptr %0, i32 0, i32 %56
  %58 = load { double, double }, ptr %57, align 8
  %59 = trunc i8 %37 to i1
  %60 = select i1 %59, { double, double } %53, { double, double } %58
  store { double, double } %60, ptr %57, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

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
!1 = !{i32 0, i32 32}
!2 = !{i32 0, i32 128}
!3 = !{}
