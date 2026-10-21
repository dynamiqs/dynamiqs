; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_select_fusion_3(ptr noalias nofree readonly align 256 captures(none) dereferenceable(1) %0, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = load i8, ptr addrspace(1) %3, align 256, !invariant.load !2
  %6 = and i8 %5, 1
  %7 = xor i8 %6, 1
  %8 = zext nneg i8 %7 to i64
  store i64 %8, ptr addrspace(1) %4, align 256
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{}
