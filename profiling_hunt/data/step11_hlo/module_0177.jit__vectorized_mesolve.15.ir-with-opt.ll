; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion_2(ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %2, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %3) local_unnamed_addr #0 {
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %0 to ptr addrspace(1)
  %7 = addrspacecast ptr %2 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = load double, ptr addrspace(1) %5, align 256, !invariant.load !2
  %10 = load double, ptr addrspace(1) %6, align 256, !invariant.load !2
  %11 = fneg double %10
  %12 = load i64, ptr addrspace(1) %7, align 256, !invariant.load !2
  %13 = icmp eq i64 %12, 1
  %14 = fneg double %9
  %15 = select i1 %13, double %9, double %11
  %16 = select i1 %13, double %10, double %14
  %17 = sitofp i64 %12 to double
  %18 = fsub double %15, %16
  %19 = fmul double %18, %17
  store double %19, ptr addrspace(1) %8, align 256
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{}
