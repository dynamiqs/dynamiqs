; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @input_concatenate_fusion(ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %0, ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(1280) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = shl nuw nsw i32 %7, 7
  %10 = or disjoint i32 %9, %8
  %11 = icmp samesign ult i32 %10, 159
  br i1 %11, label %12, label %24

12:                                               ; preds = %3
  %13 = uitofp nneg i32 %10 to double
  %14 = fmul nnan double %13, f0x3F79C2D14EE4A102
  %15 = load double, ptr addrspace(1) %4, align 16, !invariant.load !4
  %16 = fsub nnan double 1.000000e+00, %14
  %17 = load double, ptr addrspace(1) %5, align 16, !invariant.load !4
  %18 = fmul double %17, f0x3F79C2D14EE4A102
  %19 = fmul double %16, %15
  %20 = fmul double %18, %13
  %21 = fadd double %19, %20
  %22 = zext nneg i32 %10 to i64
  %23 = getelementptr inbounds [8 x i8], ptr addrspace(1) %6, i64 %22
  store double %21, ptr addrspace(1) %23, align 8
  br label %24

24:                                               ; preds = %12, %3
  %25 = icmp eq i32 %10, 0
  br i1 %25, label %26, label %28

26:                                               ; preds = %24
  %27 = load double, ptr addrspace(1) %5, align 16, !invariant.load !4
  %sunkaddr = getelementptr inbounds i8, ptr addrspace(1) %6, i64 1272
  store double %27, ptr addrspace(1) %sunkaddr, align 8
  br label %28

28:                                               ; preds = %26, %24
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 2}
!3 = !{i32 0, i32 128}
!4 = !{}
