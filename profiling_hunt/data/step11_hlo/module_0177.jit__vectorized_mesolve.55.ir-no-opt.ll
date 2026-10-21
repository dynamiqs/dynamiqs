; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [1056 x { double, double }] undef

define ptx_kernel void @input_transpose_fusion_1(ptr noalias align 256 dereferenceable(65536) %0, ptr noalias align 256 dereferenceable(65536) %1, ptr noalias align 256 dereferenceable(1) %2, ptr noalias align 256 dereferenceable(65536) %3) #0 {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %7 = urem i32 %6, 2
  %8 = mul i32 %7, 32
  %9 = udiv i32 %5, 32
  %10 = mul i32 %9, 64
  %11 = add i32 %8, %10
  %12 = udiv i32 %6, 2
  %13 = mul i32 %12, 2048
  %14 = add i32 %11, %13
  %15 = urem i32 %5, 32
  %16 = add i32 %14, %15
  %17 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %16
  %18 = load { double, double }, ptr %17, align 8, !invariant.load !3
  %19 = mul i32 %15, 33
  %20 = add i32 %19, %9
  %21 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %20
  store { double, double } %18, ptr %21, align 8
  %22 = add i32 %16, 256
  %23 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %22
  %24 = load { double, double }, ptr %23, align 8, !invariant.load !3
  %25 = add i32 %20, 4
  %26 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %25
  store { double, double } %24, ptr %26, align 8
  %27 = add i32 %16, 512
  %28 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %27
  %29 = load { double, double }, ptr %28, align 8, !invariant.load !3
  %30 = add i32 %20, 8
  %31 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %30
  store { double, double } %29, ptr %31, align 8
  %32 = add i32 %16, 768
  %33 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %32
  %34 = load { double, double }, ptr %33, align 8, !invariant.load !3
  %35 = add i32 %20, 12
  %36 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %35
  store { double, double } %34, ptr %36, align 8
  %37 = add i32 %16, 1024
  %38 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %37
  %39 = load { double, double }, ptr %38, align 8, !invariant.load !3
  %40 = add i32 %20, 16
  %41 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %40
  store { double, double } %39, ptr %41, align 8
  %42 = add i32 %16, 1280
  %43 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %42
  %44 = load { double, double }, ptr %43, align 8, !invariant.load !3
  %45 = add i32 %20, 20
  %46 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %45
  store { double, double } %44, ptr %46, align 8
  %47 = add i32 %16, 1536
  %48 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %47
  %49 = load { double, double }, ptr %48, align 8, !invariant.load !3
  %50 = add i32 %20, 24
  %51 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %50
  store { double, double } %49, ptr %51, align 8
  %52 = add i32 %16, 1792
  %53 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %52
  %54 = load { double, double }, ptr %53, align 8, !invariant.load !3
  %55 = add i32 %20, 28
  %56 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %55
  store { double, double } %54, ptr %56, align 8
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %57 = getelementptr inbounds [1 x i8], ptr %2, i32 0, i32 0
  %58 = load i8, ptr %57, align 1, !invariant.load !3
  %59 = trunc i8 %58 to i1
  %60 = mul i32 %9, 33
  %61 = add i32 %60, %15
  %62 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %61
  %63 = load { double, double }, ptr %62, align 8
  %64 = mul i32 %7, 2048
  %65 = add i32 %64, %10
  %66 = mul i32 %12, 32
  %67 = add i32 %65, %66
  %68 = add i32 %67, %15
  %69 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %68
  %70 = load { double, double }, ptr %69, align 8
  %71 = select i1 %59, { double, double } %63, { double, double } %70
  store { double, double } %71, ptr %69, align 8
  %72 = add i32 %61, 132
  %73 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %72
  %74 = load { double, double }, ptr %73, align 8
  %75 = add i32 %68, 256
  %76 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %75
  %77 = load { double, double }, ptr %76, align 8
  %78 = select i1 %59, { double, double } %74, { double, double } %77
  store { double, double } %78, ptr %76, align 8
  %79 = add i32 %61, 264
  %80 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %79
  %81 = load { double, double }, ptr %80, align 8
  %82 = add i32 %68, 512
  %83 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %82
  %84 = load { double, double }, ptr %83, align 8
  %85 = select i1 %59, { double, double } %81, { double, double } %84
  store { double, double } %85, ptr %83, align 8
  %86 = add i32 %61, 396
  %87 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %86
  %88 = load { double, double }, ptr %87, align 8
  %89 = add i32 %68, 768
  %90 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %89
  %91 = load { double, double }, ptr %90, align 8
  %92 = select i1 %59, { double, double } %88, { double, double } %91
  store { double, double } %92, ptr %90, align 8
  %93 = add i32 %61, 528
  %94 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %93
  %95 = load { double, double }, ptr %94, align 8
  %96 = add i32 %68, 1024
  %97 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %96
  %98 = load { double, double }, ptr %97, align 8
  %99 = select i1 %59, { double, double } %95, { double, double } %98
  store { double, double } %99, ptr %97, align 8
  %100 = add i32 %61, 660
  %101 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %100
  %102 = load { double, double }, ptr %101, align 8
  %103 = add i32 %68, 1280
  %104 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %103
  %105 = load { double, double }, ptr %104, align 8
  %106 = select i1 %59, { double, double } %102, { double, double } %105
  store { double, double } %106, ptr %104, align 8
  %107 = add i32 %61, 792
  %108 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %107
  %109 = load { double, double }, ptr %108, align 8
  %110 = add i32 %68, 1536
  %111 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %110
  %112 = load { double, double }, ptr %111, align 8
  %113 = select i1 %59, { double, double } %109, { double, double } %112
  store { double, double } %113, ptr %111, align 8
  %114 = add i32 %61, 924
  %115 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %114
  %116 = load { double, double }, ptr %115, align 8
  %117 = add i32 %68, 1792
  %118 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %117
  %119 = load { double, double }, ptr %118, align 8
  %120 = select i1 %59, { double, double } %116, { double, double } %119
  store { double, double } %120, ptr %118, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 128}
!2 = !{i32 0, i32 4}
!3 = !{}
