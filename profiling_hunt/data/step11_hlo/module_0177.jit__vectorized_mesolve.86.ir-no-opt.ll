; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [1056 x { double, double }] undef

define ptx_kernel void @input_transpose_fusion(ptr noalias align 256 dereferenceable(65536) %0, ptr noalias align 256 dereferenceable(65536) %1) #0 {
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
  %64 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %63
  %65 = load { double, double }, ptr %64, align 8, !invariant.load !3
  %66 = extractvalue { double, double } %65, 1
  %67 = extractvalue { double, double } %65, 0
  %68 = fneg double %66
  %69 = extractvalue { double, double } %58, 0
  %70 = fadd double %69, %67
  %71 = extractvalue { double, double } %58, 1
  %72 = fadd double %71, %68
  %73 = insertvalue { double, double } poison, double %70, 0
  %74 = insertvalue { double, double } %73, double %72, 1
  %75 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %63
  store { double, double } %74, ptr %75, align 8
  %76 = add i32 %56, 132
  %77 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %76
  %78 = load { double, double }, ptr %77, align 8
  %79 = add i32 %63, 256
  %80 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %79
  %81 = load { double, double }, ptr %80, align 8, !invariant.load !3
  %82 = extractvalue { double, double } %81, 1
  %83 = extractvalue { double, double } %81, 0
  %84 = fneg double %82
  %85 = extractvalue { double, double } %78, 0
  %86 = fadd double %85, %83
  %87 = extractvalue { double, double } %78, 1
  %88 = fadd double %87, %84
  %89 = insertvalue { double, double } poison, double %86, 0
  %90 = insertvalue { double, double } %89, double %88, 1
  %91 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %79
  store { double, double } %90, ptr %91, align 8
  %92 = add i32 %56, 264
  %93 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %92
  %94 = load { double, double }, ptr %93, align 8
  %95 = add i32 %63, 512
  %96 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %95
  %97 = load { double, double }, ptr %96, align 8, !invariant.load !3
  %98 = extractvalue { double, double } %97, 1
  %99 = extractvalue { double, double } %97, 0
  %100 = fneg double %98
  %101 = extractvalue { double, double } %94, 0
  %102 = fadd double %101, %99
  %103 = extractvalue { double, double } %94, 1
  %104 = fadd double %103, %100
  %105 = insertvalue { double, double } poison, double %102, 0
  %106 = insertvalue { double, double } %105, double %104, 1
  %107 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %95
  store { double, double } %106, ptr %107, align 8
  %108 = add i32 %56, 396
  %109 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %108
  %110 = load { double, double }, ptr %109, align 8
  %111 = add i32 %63, 768
  %112 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %111
  %113 = load { double, double }, ptr %112, align 8, !invariant.load !3
  %114 = extractvalue { double, double } %113, 1
  %115 = extractvalue { double, double } %113, 0
  %116 = fneg double %114
  %117 = extractvalue { double, double } %110, 0
  %118 = fadd double %117, %115
  %119 = extractvalue { double, double } %110, 1
  %120 = fadd double %119, %116
  %121 = insertvalue { double, double } poison, double %118, 0
  %122 = insertvalue { double, double } %121, double %120, 1
  %123 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %111
  store { double, double } %122, ptr %123, align 8
  %124 = add i32 %56, 528
  %125 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %124
  %126 = load { double, double }, ptr %125, align 8
  %127 = add i32 %63, 1024
  %128 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %127
  %129 = load { double, double }, ptr %128, align 8, !invariant.load !3
  %130 = extractvalue { double, double } %129, 1
  %131 = extractvalue { double, double } %129, 0
  %132 = fneg double %130
  %133 = extractvalue { double, double } %126, 0
  %134 = fadd double %133, %131
  %135 = extractvalue { double, double } %126, 1
  %136 = fadd double %135, %132
  %137 = insertvalue { double, double } poison, double %134, 0
  %138 = insertvalue { double, double } %137, double %136, 1
  %139 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %127
  store { double, double } %138, ptr %139, align 8
  %140 = add i32 %56, 660
  %141 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %140
  %142 = load { double, double }, ptr %141, align 8
  %143 = add i32 %63, 1280
  %144 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %143
  %145 = load { double, double }, ptr %144, align 8, !invariant.load !3
  %146 = extractvalue { double, double } %145, 1
  %147 = extractvalue { double, double } %145, 0
  %148 = fneg double %146
  %149 = extractvalue { double, double } %142, 0
  %150 = fadd double %149, %147
  %151 = extractvalue { double, double } %142, 1
  %152 = fadd double %151, %148
  %153 = insertvalue { double, double } poison, double %150, 0
  %154 = insertvalue { double, double } %153, double %152, 1
  %155 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %143
  store { double, double } %154, ptr %155, align 8
  %156 = add i32 %56, 792
  %157 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %156
  %158 = load { double, double }, ptr %157, align 8
  %159 = add i32 %63, 1536
  %160 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %159
  %161 = load { double, double }, ptr %160, align 8, !invariant.load !3
  %162 = extractvalue { double, double } %161, 1
  %163 = extractvalue { double, double } %161, 0
  %164 = fneg double %162
  %165 = extractvalue { double, double } %158, 0
  %166 = fadd double %165, %163
  %167 = extractvalue { double, double } %158, 1
  %168 = fadd double %167, %164
  %169 = insertvalue { double, double } poison, double %166, 0
  %170 = insertvalue { double, double } %169, double %168, 1
  %171 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %159
  store { double, double } %170, ptr %171, align 8
  %172 = add i32 %56, 924
  %173 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %172
  %174 = load { double, double }, ptr %173, align 8
  %175 = add i32 %63, 1792
  %176 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %175
  %177 = load { double, double }, ptr %176, align 8, !invariant.load !3
  %178 = extractvalue { double, double } %177, 1
  %179 = extractvalue { double, double } %177, 0
  %180 = fneg double %178
  %181 = extractvalue { double, double } %174, 0
  %182 = fadd double %181, %179
  %183 = extractvalue { double, double } %174, 1
  %184 = fadd double %183, %180
  %185 = insertvalue { double, double } poison, double %182, 0
  %186 = insertvalue { double, double } %185, double %184, 1
  %187 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %175
  store { double, double } %186, ptr %187, align 8
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
