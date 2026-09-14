; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@global_smem = external addrspace(3) global [0 x i8], align 16

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32) #1

define ptx_kernel void @input_reduce_fusion(ptr noalias align 16 dereferenceable(2) %arg0, ptr noalias align 256 dereferenceable(1) %arg1) #2 {
  %1 = addrspacecast ptr %arg0 to ptr addrspace(1)
  %2 = addrspacecast ptr %arg1 to ptr addrspace(1)
  %3 = addrspacecast ptr null to ptr addrspace(1)
  %4 = addrspacecast ptr null to ptr addrspace(1)
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = and i32 %5, 31
  %7 = shl i32 %6, 0
  %8 = or i32 0, %7
  %9 = and i32 %8, 1
  %10 = or disjoint i32 %9, 0
  %11 = xor i32 0, %10
  %12 = xor i32 %11, 0
  %13 = add i32 %12, 0
  %14 = sext i32 %13 to i64
  %15 = getelementptr i8, ptr addrspace(1) %1, i64 %14
  %16 = call i8 asm sideeffect "mov.u16 $0, 0x0; ld.global.b8 { $0 }, [ $1 + 0 ];", "=c,l"(ptr addrspace(1) %15)
  %17 = bitcast i8 %16 to <1 x i8>
  %18 = extractelement <1 x i8> %17, i32 0
  %19 = icmp ne i8 %18, 0
  %20 = zext i1 %19 to i32
  %21 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %20, i32 1, i32 31)
  %22 = trunc i32 %21 to i1
  %23 = or i1 %19, %22
  %24 = zext i1 %23 to i8
  %25 = and i32 %6, -1
  %26 = icmp eq i32 %25, 0
  %27 = and i1 %26, true
  %28 = and i1 %27, true
  %29 = insertelement <1 x i8> undef, i8 %24, i32 0
  %30 = bitcast <1 x i8> %29 to i8
  call void asm sideeffect "@$2 st.global.b8 [ $1 + 0 ], { $0 };", "c,l,b"(i8 %30, ptr addrspace(1) %2, i1 %28)
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #2 = { "nvvm.reqntid"="32,1,1" }

!llvm.module.flags = !{!0}
!nvvm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
