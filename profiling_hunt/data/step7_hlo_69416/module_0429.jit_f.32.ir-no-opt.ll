; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@global_smem = external addrspace(3) global [0 x i8], align 16

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32) #1

define ptx_kernel void @input_reduce_fusion(ptr noalias align 256 dereferenceable(1) %arg0) #2 {
  %1 = addrspacecast ptr %arg0 to ptr addrspace(1)
  %2 = addrspacecast ptr null to ptr addrspace(1)
  %3 = addrspacecast ptr null to ptr addrspace(1)
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %5 = and i32 %4, 31
  %6 = shl i32 %5, 0
  %7 = or i32 0, %6
  %8 = and i32 %7, 1
  %9 = or disjoint i32 %8, 0
  %10 = xor i32 0, %9
  %11 = xor i32 %10, 0
  %12 = add i32 %11, 0
  %13 = mul i32 %12, 2
  %14 = icmp sge i32 %13, 0
  %15 = zext i1 %14 to i32
  %16 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %15, i32 1, i32 31)
  %17 = trunc i32 %16 to i1
  %18 = and i1 %14, %17
  %19 = zext i1 %18 to i8
  %20 = and i32 %5, -1
  %21 = icmp eq i32 %20, 0
  %22 = and i1 %21, true
  %23 = and i1 %22, true
  %24 = insertelement <1 x i8> undef, i8 %19, i32 0
  %25 = bitcast <1 x i8> %24 to i8
  call void asm sideeffect "@$2 st.global.b8 [ $1 + 0 ], { $0 };", "c,l,b"(i8 %25, ptr addrspace(1) %1, i1 %23)
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #2 = { "nvvm.reqntid"="32,1,1" }

!llvm.module.flags = !{!0}
!nvvm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
