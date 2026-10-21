; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32) #1

; Function Attrs: nounwind
define ptx_kernel void @input_reduce_fusion(ptr noalias align 256 dereferenceable(1) %arg0) local_unnamed_addr #2 {
  %1 = addrspacecast ptr %arg0 to ptr addrspace(1)
  %2 = tail call range(i32 0, 32) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %3 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 1, i32 1, i32 31)
  %4 = trunc i32 %3 to i8
  %5 = and i8 %4, 1
  %6 = icmp eq i32 %2, 0
  tail call void asm sideeffect "@$2 st.global.b8 [ $1 + 0 ], { $0 };", "c,l,b"(i8 %5, ptr addrspace(1) %1, i1 %6) #3
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #2 = { nounwind "nvvm.reqntid"="32,1,1" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1}
!nvvm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
