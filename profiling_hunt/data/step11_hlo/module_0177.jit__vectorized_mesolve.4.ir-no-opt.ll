; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @input_reduce_fusion_4(ptr noalias align 256 dereferenceable(320) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(1) %3) #0 {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %6 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  %7 = load double, ptr %6, align 8, !invariant.load !2
  %8 = getelementptr inbounds [40 x double], ptr %0, i32 0, i32 %5
  %9 = load double, ptr %8, align 8, !invariant.load !2
  %10 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %11 = load double, ptr %10, align 8, !invariant.load !2
  %12 = fcmp ogt double %9, %7
  %13 = fcmp olt double %9, %11
  %14 = or i1 %12, %13
  %15 = zext i1 %14 to i8
  %16 = call i8 @region_3_8_clone_reduce_or_29_0(i8 0, i8 %15)
  %17 = icmp sle i32 %5, 7
  br i1 %17, label %18, label %29

18:                                               ; preds = %4
  %19 = add i32 %5, 32
  %20 = load double, ptr %6, align 8, !invariant.load !2
  %21 = getelementptr inbounds [40 x double], ptr %0, i32 0, i32 %19
  %22 = load double, ptr %21, align 8, !invariant.load !2
  %23 = load double, ptr %10, align 8, !invariant.load !2
  %24 = fcmp ogt double %22, %20
  %25 = fcmp olt double %22, %23
  %26 = or i1 %24, %25
  %27 = zext i1 %26 to i8
  %28 = call i8 @region_3_8_clone_reduce_or_29_0(i8 %16, i8 %27)
  br label %30

29:                                               ; preds = %4
  br label %30

30:                                               ; preds = %18, %29
  %31 = phi i8 [ %16, %29 ], [ %28, %18 ]
  br label %32

32:                                               ; preds = %30
  %33 = zext i8 %31 to i32
  %34 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %33, i32 16, i32 31)
  %35 = trunc i32 %34 to i8
  %36 = call i8 @region_3_8_clone_reduce_or_29_0(i8 %31, i8 %35)
  %37 = zext i8 %36 to i32
  %38 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %37, i32 8, i32 31)
  %39 = trunc i32 %38 to i8
  %40 = call i8 @region_3_8_clone_reduce_or_29_0(i8 %36, i8 %39)
  %41 = zext i8 %40 to i32
  %42 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %41, i32 4, i32 31)
  %43 = trunc i32 %42 to i8
  %44 = call i8 @region_3_8_clone_reduce_or_29_0(i8 %40, i8 %43)
  %45 = zext i8 %44 to i32
  %46 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %45, i32 2, i32 31)
  %47 = trunc i32 %46 to i8
  %48 = call i8 @region_3_8_clone_reduce_or_29_0(i8 %44, i8 %47)
  %49 = zext i8 %48 to i32
  %50 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %49, i32 1, i32 31)
  %51 = trunc i32 %50 to i8
  %52 = call i8 @region_3_8_clone_reduce_or_29_0(i8 %48, i8 %51)
  %53 = icmp eq i32 %5, 0
  br i1 %53, label %54, label %56

54:                                               ; preds = %32
  %55 = getelementptr inbounds [1 x i8], ptr %3, i32 0, i32 0
  store i8 %52, ptr %55, align 1
  br label %56

56:                                               ; preds = %54, %32
  ret void
}

define internal i8 @region_3_8_clone_reduce_or_29_0(i8 %0, i8 %1) {
  %3 = or i8 %0, %1
  ret i8 %3
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #2

attributes #0 = { "nvvm.reqntid"="32,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 32}
!2 = !{}
