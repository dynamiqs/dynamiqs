; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @input_reduce_fusion_4(ptr noalias nofree readonly align 256 captures(none) dereferenceable(320) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %2, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(1) %3) local_unnamed_addr #0 {
  %5 = addrspacecast ptr %2 to ptr addrspace(1)
  %6 = addrspacecast ptr %0 to ptr addrspace(1)
  %7 = addrspacecast ptr %1 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %10 = load double, ptr addrspace(1) %5, align 256, !invariant.load !3
  %11 = zext nneg i32 %9 to i64
  %12 = getelementptr inbounds [8 x i8], ptr addrspace(1) %6, i64 %11
  %13 = load double, ptr addrspace(1) %12, align 8, !invariant.load !3
  %14 = load double, ptr addrspace(1) %7, align 256, !invariant.load !3
  %15 = fcmp ogt double %13, %10
  %16 = fcmp olt double %13, %14
  %17 = or i1 %15, %16
  %18 = icmp samesign ult i32 %9, 8
  br i1 %18, label %19, label %26

19:                                               ; preds = %4
  %20 = getelementptr inbounds i8, ptr addrspace(1) %12, i64 256
  %21 = load double, ptr addrspace(1) %20, align 8, !invariant.load !3
  %22 = fcmp ogt double %21, %10
  %23 = fcmp olt double %21, %14
  %24 = or i1 %22, %23
  %25 = or i1 %17, %24
  br label %26

26:                                               ; preds = %19, %4
  %.in = phi i1 [ %25, %19 ], [ %17, %4 ]
  %27 = zext i1 %.in to i8
  %28 = zext i1 %.in to i32
  %29 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %28, i32 16, i32 31)
  %30 = trunc i32 %29 to i8
  %31 = or i8 %30, %27
  %32 = zext i8 %31 to i32
  %33 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %32, i32 8, i32 31)
  %34 = trunc i32 %33 to i8
  %35 = or i8 %31, %34
  %36 = zext i8 %35 to i32
  %37 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %36, i32 4, i32 31)
  %38 = trunc i32 %37 to i8
  %39 = or i8 %35, %38
  %40 = zext i8 %39 to i32
  %41 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %40, i32 2, i32 31)
  %42 = trunc i32 %41 to i8
  %43 = or i8 %39, %42
  %44 = zext i8 %43 to i32
  %45 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %44, i32 1, i32 31)
  %46 = icmp eq i32 %9, 0
  %47 = trunc i32 %45 to i8
  %48 = or i8 %43, %47
  br i1 %46, label %49, label %50

49:                                               ; preds = %26
  store i8 %48, ptr addrspace(1) %8, align 256
  br label %50

50:                                               ; preds = %49, %26
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #2

attributes #0 = { norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) "nvvm.reqntid"="32,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 32}
!3 = !{}
