; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32) #1

; Function Attrs: nounwind
define ptx_kernel void @input_reduce_fusion(ptr noalias align 16 dereferenceable(2) %arg0, ptr noalias align 256 dereferenceable(1) %arg1) local_unnamed_addr #2 {
  %1 = addrspacecast ptr %arg0 to ptr addrspace(1)
  %2 = addrspacecast ptr %arg1 to ptr addrspace(1)
  %3 = tail call range(i32 0, 32) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %4 = and i32 %3, 1
  %5 = zext nneg i32 %4 to i64
  %6 = getelementptr i8, ptr addrspace(1) %1, i64 %5
  %7 = tail call i8 asm sideeffect "mov.u16 $0, 0x0; ld.global.b8 { $0 }, [ $1 + 0 ];", "=c,l"(ptr addrspace(1) %6) #3
  %8 = icmp ne i8 %7, 0
  %9 = zext i1 %8 to i32
  %10 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %9, i32 1, i32 31)
  %11 = trunc i32 %10 to i1
  %12 = or i1 %8, %11
  %13 = zext i1 %12 to i8
  %14 = icmp eq i32 %3, 0
  tail call void asm sideeffect "@$2 st.global.b8 [ $1 + 0 ], { $0 };", "c,l,b"(i8 %13, ptr addrspace(1) %2, i1 %14) #3
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
