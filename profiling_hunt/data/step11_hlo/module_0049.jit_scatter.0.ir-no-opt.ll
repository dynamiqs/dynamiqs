; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @input_reduce_fusion(ptr noalias align 16 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(1) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = zext i32 %3 to i64
  %5 = trunc i64 %4 to i32
  %6 = icmp ult i32 %3, 1
  br i1 %6, label %7, label %10

7:                                                ; preds = %2
  %8 = getelementptr inbounds [2 x i32], ptr %0, i32 0, i32 0
  %9 = load i32, ptr %8, align 4, !invariant.load !2
  br label %13

10:                                               ; preds = %2
  %11 = getelementptr inbounds [2 x i32], ptr %0, i32 0, i32 1
  %12 = load i32, ptr %11, align 4, !invariant.load !2
  br label %13

13:                                               ; preds = %7, %10
  %14 = phi i32 [ %12, %10 ], [ %9, %7 ]
  br label %15

15:                                               ; preds = %13
  %16 = icmp sge i32 %14, 0
  %17 = icmp sge i32 %5, %14
  %18 = and i1 %16, %17
  %19 = zext i1 %18 to i8
  %20 = call i8 @and_reduce_sub_computation_and_3(i8 1, i8 %19)
  %21 = zext i8 %20 to i32
  %22 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %21, i32 1, i32 31)
  %23 = trunc i32 %22 to i8
  %24 = call i8 @and_reduce_sub_computation_and_3(i8 %20, i8 %23)
  %25 = icmp eq i32 %3, 0
  br i1 %25, label %26, label %28

26:                                               ; preds = %15
  %27 = getelementptr inbounds [1 x i8], ptr %1, i32 0, i32 0
  store i8 %24, ptr %27, align 1
  br label %28

28:                                               ; preds = %26, %15
  ret void
}

define internal i8 @and_reduce_sub_computation_and_3(i8 %0, i8 %1) {
  %3 = and i8 %0, %1
  ret i8 %3
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #2

attributes #0 = { "nvvm.reqntid"="2,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 2}
!2 = !{}
