; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define ptx_kernel void @wrapped_iota(ptr noalias nofree writeonly align 256 captures(none) dereferenceable(1008) %0) local_unnamed_addr #0 {
  %2 = addrspacecast ptr %0 to ptr addrspace(1)
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %4 = uitofp nneg i32 %3 to double
  %5 = zext nneg i32 %3 to i64
  %6 = getelementptr inbounds [16 x i8], ptr addrspace(1) %2, i64 %5
  %7 = insertelement <2 x double> poison, double %4, i32 0
  %8 = insertelement <2 x double> %7, double 0.000000e+00, i32 1
  store <2 x double> %8, ptr addrspace(1) %6, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) "nvvm.reqntid"="63,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 63}
