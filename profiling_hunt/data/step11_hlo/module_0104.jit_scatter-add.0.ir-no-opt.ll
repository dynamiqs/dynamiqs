; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @input_reduce_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 16 dereferenceable(4) %1, ptr noalias align 256 dereferenceable(1) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %5 = getelementptr inbounds [2 x i32], ptr %0, i32 0, i32 %4
  %6 = load i32, ptr %5, align 4, !invariant.load !2
  %7 = icmp ult i32 %4, 1
  br i1 %7, label %8, label %11

8:                                                ; preds = %3
  %9 = getelementptr inbounds [1 x i32], ptr %1, i32 0, i32 %4
  %10 = load i32, ptr %9, align 4, !invariant.load !2
  br label %12

11:                                               ; preds = %3
  br label %12

12:                                               ; preds = %8, %11
  %13 = phi i32 [ 0, %11 ], [ %10, %8 ]
  br label %14

14:                                               ; preds = %12
  %15 = icmp sge i32 %13, 0
  %16 = icmp sge i32 %6, %13
  %17 = and i1 %15, %16
  %18 = zext i1 %17 to i8
  %19 = call i8 @and_reduce_sub_computation_and_6(i8 1, i8 %18)
  %20 = zext i8 %19 to i32
  %21 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %20, i32 1, i32 31)
  %22 = trunc i32 %21 to i8
  %23 = call i8 @and_reduce_sub_computation_and_6(i8 %19, i8 %22)
  %24 = icmp eq i32 %4, 0
  br i1 %24, label %25, label %27

25:                                               ; preds = %14
  %26 = getelementptr inbounds [1 x i8], ptr %2, i32 0, i32 0
  store i8 %23, ptr %26, align 1
  br label %27

27:                                               ; preds = %25, %14
  ret void
}

define internal i8 @and_reduce_sub_computation_and_6(i8 %0, i8 %1) {
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
