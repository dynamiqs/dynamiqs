; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@global_smem = external addrspace(3) global [0 x i8], align 16

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.redux.sync.or(i32, i32) #1

define ptx_kernel void @input_reduce_fusion_3(ptr noalias align 256 dereferenceable(320) %arg0, ptr noalias align 256 dereferenceable(1) %arg1) #2 {
  %1 = addrspacecast ptr %arg0 to ptr addrspace(1)
  %2 = addrspacecast ptr %arg1 to ptr addrspace(1)
  %3 = addrspacecast ptr null to ptr addrspace(1)
  %4 = addrspacecast ptr null to ptr addrspace(1)
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = and i32 %5, 31
  %7 = shl i32 %6, 0
  %8 = or i32 0, %7
  %9 = and i32 %8, 31
  %10 = shl i32 %9, 1
  %11 = or disjoint i32 %10, 0
  %12 = xor i32 0, %11
  %13 = xor i32 %12, 0
  %14 = xor i32 %12, 1
  %15 = add i32 %13, 0
  %16 = add i32 %14, 0
  %17 = sext i32 %15 to i64
  %18 = sext i32 %16 to i64
  %19 = icmp slt i64 %17, 39
  %20 = icmp slt i64 %18, 39
  %21 = getelementptr double, ptr addrspace(1) %1, i32 1
  %22 = getelementptr double, ptr addrspace(1) %21, i64 %17
  %23 = getelementptr double, ptr addrspace(1) %21, i64 %18
  %24 = call i64 asm sideeffect "mov.u64 $0, $1; @$3 ld.global.b64 { $0 }, [ $2 + 0 ];", "=l,l,l,b"(i64 0, ptr addrspace(1) %22, i1 %19)
  %25 = bitcast i64 %24 to <1 x double>
  %26 = extractelement <1 x double> %25, i32 0
  %27 = call i64 asm sideeffect "mov.u64 $0, $1; @$3 ld.global.b64 { $0 }, [ $2 + 0 ];", "=l,l,l,b"(i64 0, ptr addrspace(1) %23, i1 %20)
  %28 = bitcast i64 %27 to <1 x double>
  %29 = extractelement <1 x double> %28, i32 0
  %30 = icmp slt i64 %17, 40
  %31 = getelementptr double, ptr addrspace(1) %1, i64 %17
  %32 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2; mov.u64 $1, $3; @$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %31, i1 %30)
  %33 = extractvalue { i64, i64 } %32, 0
  %34 = bitcast i64 %33 to <1 x double>
  %35 = extractvalue { i64, i64 } %32, 1
  %36 = bitcast i64 %35 to <1 x double>
  %37 = extractelement <1 x double> %34, i32 0
  %38 = extractelement <1 x double> %36, i32 0
  %39 = fcmp olt double %26, %37
  %40 = fcmp olt double %29, %38
  %41 = icmp slt i32 %15, 39
  %42 = icmp slt i32 %16, 39
  %43 = select i1 %41, i1 %39, i1 false
  %44 = select i1 %42, i1 %40, i1 false
  %45 = or i1 %43, %44
  %46 = zext i1 %45 to i32
  %47 = call i32 @llvm.nvvm.redux.sync.or(i32 %46, i32 -1)
  %48 = trunc i32 %47 to i1
  %49 = zext i1 %48 to i8
  %50 = and i32 %6, -1
  %51 = icmp eq i32 %50, 0
  %52 = and i1 %51, true
  %53 = and i1 %52, true
  %54 = insertelement <1 x i8> undef, i8 %49, i32 0
  %55 = bitcast <1 x i8> %54 to i8
  call void asm sideeffect "@$2 st.global.b8 [ $1 + 0 ], { $0 };", "c,l,b"(i8 %55, ptr addrspace(1) %2, i1 %53)
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #2 = { "nvvm.reqntid"="32,1,1" }

!llvm.module.flags = !{!0}
!nvvm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
