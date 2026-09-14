; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @input_reduce_fusion(ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %0, ptr noalias nofree readonly align 16 captures(none) dereferenceable(4) %1, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(1) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %8 = zext nneg i32 %7 to i64
  %9 = getelementptr inbounds [4 x i8], ptr addrspace(1) %4, i64 %8
  %10 = load i32, ptr addrspace(1) %9, align 4, !invariant.load !3
  %.not.not = icmp eq i32 %7, 0
  %11 = load i32, ptr addrspace(1) %5, align 16, !invariant.load !3
  %spec.select = select i1 %.not.not, i32 %11, i32 0
  %12 = icmp sgt i32 %spec.select, -1
  %13 = icmp sge i32 %10, %spec.select
  %14 = and i1 %12, %13
  %15 = zext i1 %14 to i32
  %16 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %15, i32 1, i32 31)
  %17 = trunc i32 %16 to i8
  %18 = and i8 %17, 1
  %19 = select i1 %14, i8 %18, i8 0
  br i1 %.not.not, label %20, label %21

20:                                               ; preds = %3
  store i8 %19, ptr addrspace(1) %6, align 256
  br label %21

21:                                               ; preds = %20, %3
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #2

attributes #0 = { norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) "nvvm.reqntid"="2,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 2}
!3 = !{}
