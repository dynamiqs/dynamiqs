; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_pad_fusion(ptr noalias nofree readonly align 16 captures(none) dereferenceable(1008) %0, ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(1024) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %.not = icmp eq i32 %7, 0
  br i1 %.not, label %13, label %8

8:                                                ; preds = %3
  %9 = zext nneg i32 %7 to i64
  %10 = getelementptr [16 x i8], ptr addrspace(1) %4, i64 %9
  %11 = getelementptr i8, ptr addrspace(1) %10, i64 -16
  %12 = load <2 x double>, ptr addrspace(1) %11, align 16, !invariant.load !3
  %.unpack8 = extractelement <2 x double> %12, i32 0
  %.unpack29 = extractelement <2 x double> %12, i32 1
  br label %16

13:                                               ; preds = %3
  %14 = load i64, ptr addrspace(1) %5, align 16, !invariant.load !3
  %15 = sitofp i64 %14 to double
  br label %16

16:                                               ; preds = %13, %8
  %.pre-phi = phi i64 [ 0, %13 ], [ %9, %8 ]
  %.pn5 = phi double [ %15, %13 ], [ %.unpack8, %8 ]
  %.pn3 = phi double [ 0.000000e+00, %13 ], [ %.unpack29, %8 ]
  %17 = getelementptr inbounds [16 x i8], ptr addrspace(1) %6, i64 %.pre-phi
  %18 = insertelement <2 x double> poison, double %.pn5, i32 0
  %19 = insertelement <2 x double> %18, double %.pn3, i32 1
  store <2 x double> %19, ptr addrspace(1) %17, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="64,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 64}
!3 = !{}
