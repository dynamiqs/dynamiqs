; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_multiply(ptr noalias nofree readonly align 16 captures(none) dereferenceable(1008) %0, ptr noalias nofree readonly align 16 captures(none) dereferenceable(1008) %1, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(1008) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %8 = zext nneg i32 %7 to i64
  %9 = getelementptr inbounds [16 x i8], ptr addrspace(1) %4, i64 %8
  %10 = load <2 x double>, ptr addrspace(1) %9, align 16, !invariant.load !3
  %.unpack8 = extractelement <2 x double> %10, i32 0
  %.unpack29 = extractelement <2 x double> %10, i32 1
  %11 = getelementptr inbounds [16 x i8], ptr addrspace(1) %5, i64 %8
  %12 = load <2 x double>, ptr addrspace(1) %11, align 16, !invariant.load !3
  %.unpack310 = extractelement <2 x double> %12, i32 0
  %.unpack511 = extractelement <2 x double> %12, i32 1
  %13 = fmul double %.unpack8, %.unpack310
  %14 = fmul double %.unpack29, %.unpack511
  %15 = fmul double %.unpack29, %.unpack310
  %16 = fmul double %.unpack8, %.unpack511
  %17 = fsub double %13, %14
  %18 = fadd double %15, %16
  %19 = getelementptr inbounds [16 x i8], ptr addrspace(1) %6, i64 %8
  %20 = insertelement <2 x double> poison, double %17, i32 0
  %21 = insertelement <2 x double> %20, double %18, i32 1
  store <2 x double> %21, ptr addrspace(1) %19, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="63,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 63}
!3 = !{}
