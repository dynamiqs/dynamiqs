; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_select(ptr noalias nofree readonly align 16 captures(none) dereferenceable(1) %0, ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %2, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %3) local_unnamed_addr #0 {
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %1 to ptr addrspace(1)
  %7 = addrspacecast ptr %2 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = load i8, ptr addrspace(1) %5, align 16, !invariant.load !2
  %10 = load i64, ptr addrspace(1) %6, align 16, !invariant.load !2
  %11 = load i64, ptr addrspace(1) %7, align 16, !invariant.load !2
  %12 = trunc i8 %9 to i1
  %13 = select i1 %12, i64 %10, i64 %11
  store i64 %13, ptr addrspace(1) %8, align 256
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{}
