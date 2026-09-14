; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @input_concatenate_fusion(ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %0, ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(640) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %8 = uitofp nneg i32 %7 to double
  %9 = fmul nnan double %8, f0x3F89EC8E951033D9
  %10 = load double, ptr addrspace(1) %4, align 16, !invariant.load !3
  %11 = fsub nnan double 1.000000e+00, %9
  %12 = load double, ptr addrspace(1) %5, align 16, !invariant.load !3
  %13 = fmul double %12, f0x3F89EC8E951033D9
  %14 = fmul double %11, %10
  %15 = fmul double %13, %8
  %16 = fadd double %14, %15
  %17 = zext nneg i32 %7 to i64
  %18 = getelementptr inbounds [8 x i8], ptr addrspace(1) %6, i64 %17
  store double %16, ptr addrspace(1) %18, align 8
  %19 = icmp eq i32 %7, 0
  br i1 %19, label %20, label %21

20:                                               ; preds = %3
  %sunkaddr = getelementptr inbounds i8, ptr addrspace(1) %6, i64 632
  store double %12, ptr addrspace(1) %sunkaddr, align 8
  br label %21

21:                                               ; preds = %20, %3
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="79,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 79}
!3 = !{}
