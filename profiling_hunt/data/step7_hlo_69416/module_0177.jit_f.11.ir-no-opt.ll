; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [1056 x { double, double }] undef

define ptx_kernel void @wrapped_transpose(ptr noalias align 256 dereferenceable(65536) %0, ptr noalias align 256 dereferenceable(65536) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %5 = urem i32 %4, 2
  %6 = mul i32 %5, 32
  %7 = udiv i32 %3, 32
  %8 = mul i32 %7, 64
  %9 = add i32 %6, %8
  %10 = udiv i32 %4, 2
  %11 = mul i32 %10, 2048
  %12 = add i32 %9, %11
  %13 = urem i32 %3, 32
  %14 = add i32 %12, %13
  %15 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %14
  %16 = load { double, double }, ptr %15, align 8, !invariant.load !3
  %17 = mul i32 %13, 33
  %18 = add i32 %17, %7
  %19 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %18
  store { double, double } %16, ptr %19, align 8
  %20 = add i32 %14, 256
  %21 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %20
  %22 = load { double, double }, ptr %21, align 8, !invariant.load !3
  %23 = add i32 %18, 4
  %24 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %23
  store { double, double } %22, ptr %24, align 8
  %25 = add i32 %14, 512
  %26 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %25
  %27 = load { double, double }, ptr %26, align 8, !invariant.load !3
  %28 = add i32 %18, 8
  %29 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %28
  store { double, double } %27, ptr %29, align 8
  %30 = add i32 %14, 768
  %31 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %30
  %32 = load { double, double }, ptr %31, align 8, !invariant.load !3
  %33 = add i32 %18, 12
  %34 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %33
  store { double, double } %32, ptr %34, align 8
  %35 = add i32 %14, 1024
  %36 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %35
  %37 = load { double, double }, ptr %36, align 8, !invariant.load !3
  %38 = add i32 %18, 16
  %39 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %38
  store { double, double } %37, ptr %39, align 8
  %40 = add i32 %14, 1280
  %41 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %40
  %42 = load { double, double }, ptr %41, align 8, !invariant.load !3
  %43 = add i32 %18, 20
  %44 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %43
  store { double, double } %42, ptr %44, align 8
  %45 = add i32 %14, 1536
  %46 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %45
  %47 = load { double, double }, ptr %46, align 8, !invariant.load !3
  %48 = add i32 %18, 24
  %49 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %48
  store { double, double } %47, ptr %49, align 8
  %50 = add i32 %14, 1792
  %51 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %50
  %52 = load { double, double }, ptr %51, align 8, !invariant.load !3
  %53 = add i32 %18, 28
  %54 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %53
  store { double, double } %52, ptr %54, align 8
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %55 = mul i32 %7, 33
  %56 = add i32 %55, %13
  %57 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %56
  %58 = load { double, double }, ptr %57, align 8
  %59 = mul i32 %5, 2048
  %60 = add i32 %59, %8
  %61 = mul i32 %10, 32
  %62 = add i32 %60, %61
  %63 = add i32 %62, %13
  %64 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %63
  store { double, double } %58, ptr %64, align 8
  %65 = add i32 %56, 132
  %66 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %65
  %67 = load { double, double }, ptr %66, align 8
  %68 = add i32 %63, 256
  %69 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %68
  store { double, double } %67, ptr %69, align 8
  %70 = add i32 %56, 264
  %71 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %70
  %72 = load { double, double }, ptr %71, align 8
  %73 = add i32 %63, 512
  %74 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %73
  store { double, double } %72, ptr %74, align 8
  %75 = add i32 %56, 396
  %76 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %75
  %77 = load { double, double }, ptr %76, align 8
  %78 = add i32 %63, 768
  %79 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %78
  store { double, double } %77, ptr %79, align 8
  %80 = add i32 %56, 528
  %81 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %80
  %82 = load { double, double }, ptr %81, align 8
  %83 = add i32 %63, 1024
  %84 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %83
  store { double, double } %82, ptr %84, align 8
  %85 = add i32 %56, 660
  %86 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %85
  %87 = load { double, double }, ptr %86, align 8
  %88 = add i32 %63, 1280
  %89 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %88
  store { double, double } %87, ptr %89, align 8
  %90 = add i32 %56, 792
  %91 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %90
  %92 = load { double, double }, ptr %91, align 8
  %93 = add i32 %63, 1536
  %94 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %93
  store { double, double } %92, ptr %94, align 8
  %95 = add i32 %56, 924
  %96 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %95
  %97 = load { double, double }, ptr %96, align 8
  %98 = add i32 %63, 1792
  %99 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %98
  store { double, double } %97, ptr %99, align 8
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
