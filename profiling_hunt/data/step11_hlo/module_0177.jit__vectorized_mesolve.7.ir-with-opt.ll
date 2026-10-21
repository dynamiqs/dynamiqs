; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.redux.sync.or(i32, i32) #1

; Function Attrs: nounwind
define ptx_kernel void @input_reduce_fusion_3(ptr noalias align 256 dereferenceable(320) %arg0, ptr noalias align 256 dereferenceable(1) %arg1) local_unnamed_addr #2 {
  %1 = addrspacecast ptr %arg0 to ptr addrspace(1)
  %2 = addrspacecast ptr %arg1 to ptr addrspace(1)
  %3 = tail call range(i32 0, 32) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %4 = shl i32 %3, 1
  %5 = or disjoint i32 %4, 1
  %6 = zext i32 %4 to i64
  %7 = icmp samesign ult i32 %3, 20
  %8 = icmp samesign ult i32 %5, 39
  %9 = getelementptr [8 x i8], ptr addrspace(1) %1, i64 %6
  %10 = getelementptr i8, ptr addrspace(1) %9, i64 8
  %11 = getelementptr i8, ptr addrspace(1) %9, i64 16
  %12 = tail call i64 asm sideeffect "mov.u64 $0, $1; @$3 ld.global.b64 { $0 }, [ $2 + 0 ];", "=l,l,l,b"(i64 0, ptr addrspace(1) %10, i1 %7) #3
  %13 = bitcast i64 %12 to double
  %14 = tail call i64 asm sideeffect "mov.u64 $0, $1; @$3 ld.global.b64 { $0 }, [ $2 + 0 ];", "=l,l,l,b"(i64 0, ptr addrspace(1) %11, i1 %8) #3
  %15 = bitcast i64 %14 to double
  %16 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2; mov.u64 $1, $3; @$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %9, i1 %7) #3
  %17 = extractvalue { i64, i64 } %16, 0
  %18 = extractvalue { i64, i64 } %16, 1
  %19 = bitcast i64 %17 to double
  %20 = bitcast i64 %18 to double
  %21 = fcmp olt double %13, %19
  %22 = fcmp olt double %15, %20
  %23 = select i1 %7, i1 %21, i1 false
  %24 = select i1 %8, i1 %22, i1 false
  %25 = or i1 %23, %24
  %26 = zext i1 %25 to i32
  %27 = tail call i32 @llvm.nvvm.redux.sync.or(i32 %26, i32 -1)
  %28 = trunc i32 %27 to i8
  %29 = and i8 %28, 1
  %30 = icmp eq i32 %3, 0
  tail call void asm sideeffect "@$2 st.global.b8 [ $1 + 0 ], { $0 };", "c,l,b"(i8 %29, ptr addrspace(1) %2, i1 %30) #3
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
