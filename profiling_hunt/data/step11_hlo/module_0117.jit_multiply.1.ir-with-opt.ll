; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion(ptr noalias nofree readonly align 16 captures(none) dereferenceable(2048) %0, ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(2048) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %8 = load double, ptr addrspace(1) %4, align 16, !invariant.load !3
  %9 = zext nneg i32 %7 to i64
  %10 = getelementptr inbounds [16 x i8], ptr addrspace(1) %5, i64 %9
  %11 = load <2 x double>, ptr addrspace(1) %10, align 16, !invariant.load !3
  %.unpack5 = extractelement <2 x double> %11, i32 0
  %.unpack26 = extractelement <2 x double> %11, i32 1
  %12 = fmul double %8, %.unpack5
  %13 = fmul double %.unpack26, 0.000000e+00
  %14 = fmul double %.unpack5, 0.000000e+00
  %15 = fmul double %8, %.unpack26
  %16 = fsub double %12, %13
  %17 = fadd double %14, %15
  %18 = getelementptr inbounds [16 x i8], ptr addrspace(1) %6, i64 %9
  %19 = insertelement <2 x double> poison, double %16, i32 0
  %20 = insertelement <2 x double> %19, double %17, i32 1
  store <2 x double> %20, ptr addrspace(1) %18, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 128}
!3 = !{}
