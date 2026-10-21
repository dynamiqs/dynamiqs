; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare double @__nv_log1p(double)

declare double @__nv_sqrt(double)

declare double @__nv_fabs(double)

define ptx_kernel void @loop_multiply_fusion(ptr noalias align 256 dereferenceable(32768) %0, ptr noalias align 256 dereferenceable(32768) %1, ptr noalias align 256 dereferenceable(65536) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %4, 128
  %7 = add i32 %6, %5
  %8 = getelementptr inbounds [4096 x double], ptr %1, i32 0, i32 %7
  %9 = load double, ptr %8, align 8, !invariant.load !3
  %10 = fneg double %9
  %11 = fmul double %9, %10
  %12 = call double @__nv_log1p(double %11)
  %13 = fneg double %12
  %14 = fcmp olt double %13, 6.250000e+00
  %15 = select i1 %14, double f0xBBB135D2E746E627, double f0x3E23040F87DBD932
  %16 = select i1 %14, double f0xBC08DDF93324D327, double f0x3E785CBE52878635
  %17 = fcmp olt double %13, 1.600000e+01
  %18 = select i1 %17, double %15, double f0xBDBDCEC3A7785389
  %19 = select i1 %17, double %16, double f0xBDF18FEEC0E38727
  %20 = call double @__nv_sqrt(double %13)
  %21 = select i1 %17, double 3.250000e+00, double 5.000000e+00
  %22 = fadd double %13, -3.125000e+00
  %23 = fsub double %20, %21
  %24 = select i1 %14, double %22, double %23
  %25 = fmul double %18, %24
  %26 = select i1 %14, double f0x3C37B83EEF0B7C9F, double f0xBE92777453DD3955
  %27 = fadd double %19, %25
  %28 = select i1 %17, double %26, double f0x3E19E6BF2DDA45E3
  %29 = fmul double %27, %24
  %30 = select i1 %14, double f0x3C69BA72CD589B91, double f0x3E5395ABCD554C6C
  %31 = fadd double %28, %29
  %32 = select i1 %17, double %30, double f0xBE30468FB24E2F5F
  %33 = fmul double %31, %24
  %34 = select i1 %14, double f0xBCA33689090A6B96, double f0x3EB936388A3790AD
  %35 = fadd double %32, %33
  %36 = select i1 %17, double %34, double f0x3E405AC6A8FBA182
  %37 = fmul double %35, %24
  %38 = select i1 %14, double f0x3C782E11898132E0, double f0xBED0D5DB812B5083
  %39 = fadd double %36, %37
  %40 = select i1 %17, double %38, double f0xBE50102E495FB9C0
  %41 = fmul double %39, %24
  %42 = select i1 %14, double f0x3CFDE4ACFD9E26BA, double f0x3EC8860CD5D652F6
  %43 = fadd double %40, %41
  %44 = select i1 %17, double %42, double f0x3E5F4C20E1334AF8
  %45 = fmul double %43, %24
  %46 = select i1 %14, double f0xBD26D33EED66C487, double f0x3EEA29A0CACDFB23
  %47 = fadd double %44, %45
  %48 = select i1 %17, double %46, double f0xBE722D220FDF9C3E
  %49 = fmul double %47, %24
  %50 = select i1 %14, double f0xBD36F2167040D8E2, double f0xBF08CEF1F80281F2
  %51 = fadd double %48, %49
  %52 = select i1 %17, double %50, double f0x3E8EBC8BB824CB54
  %53 = fmul double %51, %24
  %54 = select i1 %14, double f0x3D872A22C2D77E20, double f0x3F11E684D0B9188A
  %55 = fadd double %52, %53
  %56 = select i1 %17, double %54, double f0xBEB0A8D40EA372CC
  %57 = fmul double %55, %24
  %58 = select i1 %14, double f0xBDAC8859C4E5C0AF, double f0x3EF932CD54C8A222
  %59 = fadd double %56, %57
  %60 = select i1 %17, double %58, double f0x3ED2FBD29D093D2B
  %61 = fmul double %59, %24
  %62 = select i1 %14, double f0xBDCDC583D118A561, double f0xBF37448A89EF8AA3
  %63 = fadd double %60, %61
  %64 = select i1 %17, double %62, double f0xBEF4A3497E1E0FAC
  %65 = fmul double %63, %24
  %66 = select i1 %14, double f0x3E120F47CCF46B3C, double f0x3F4F3CC55AD40C25
  %67 = fadd double %64, %65
  %68 = select i1 %17, double %66, double f0x3F13EBF4EB00938F
  %69 = fmul double %67, %24
  %70 = select i1 %14, double f0xBE31A9E38DC84D60, double f0xBF5BA924132F38B1
  %71 = fadd double %68, %69
  %72 = select i1 %17, double %70, double f0xBF2C2F36A8FC5D53
  %73 = fmul double %71, %24
  %74 = select i1 %14, double f0xBE5F36CD6D3D46A9, double f0x3F6468EECA533CF8
  %75 = fadd double %72, %73
  %76 = select i1 %17, double %74, double f0xBF222EA5DF04047C
  %77 = fmul double %75, %24
  %78 = select i1 %14, double f0x3E9C6B4F5D03B787, double f0xBF6EBADABB891BBD
  %79 = fadd double %76, %77
  %80 = select i1 %17, double %78, double f0x3FF02A30D1FBA0DC
  %81 = fmul double %79, %24
  %82 = select i1 %14, double f0xBEB6E8A5434AE8A2, double f0x3F75FFCFE5B76AFC
  %83 = fadd double %80, %81
  %84 = select i1 %14, double f0xBEED1D1F7B8736F6, double f0x3FF0158A6D641D39
  %85 = select i1 %17, double %82, double f0x4013664DDD1AD7FB
  %86 = fmul double %83, %24
  %87 = fadd double %85, %86
  %88 = fmul double %87, %24
  %89 = select i1 %14, double f0x3F2879C2A212F024, double f0x4008ABCC380D5A48
  %90 = fadd double %84, %88
  %91 = select i1 %17, double %90, double %87
  %92 = fmul double %91, %24
  %93 = fadd double %89, %92
  %94 = select i1 %17, double %93, double %87
  %95 = fmul double %94, %24
  %96 = fadd double %95, f0xBF4845769484FCA8
  %97 = select i1 %14, double %96, double %94
  %98 = fmul double %97, %24
  %99 = fadd double %98, f0xBF78B6C33114F909
  %100 = select i1 %14, double %99, double %94
  %101 = fmul double %100, %24
  %102 = fadd double %101, f0x3FCEBD80D9B13E28
  %103 = select i1 %14, double %102, double %94
  %104 = fmul double %103, %24
  %105 = fadd double %104, f0x3FFA755E7C99AE86
  %106 = call double @__nv_fabs(double %9)
  %107 = select i1 %14, double %105, double %94
  %108 = fcmp oeq double %106, 1.000000e+00
  %109 = fmul double %9, +inf
  %110 = fmul double %107, %9
  %111 = select i1 %108, double %109, double %110
  %112 = fmul double %111, f0x3FF6A09E667F3BCD
  %113 = getelementptr inbounds [4096 x double], ptr %0, i32 0, i32 %7
  %114 = load double, ptr %113, align 8, !invariant.load !3
  %115 = fmul double %112, 0.000000e+00
  %116 = fadd double %112, 0.000000e+00
  %117 = fadd double %114, %115
  %118 = fadd double %116, 0.000000e+00
  %119 = fmul double %117, f0x3FE6A09E667F3BCC
  %120 = fmul double %118, 0.000000e+00
  %121 = fmul double %118, f0x3FE6A09E667F3BCC
  %122 = fmul double %117, 0.000000e+00
  %123 = fsub double %119, %120
  %124 = fadd double %121, %122
  %125 = insertvalue { double, double } poison, double %123, 0
  %126 = insertvalue { double, double } %125, double %124, 1
  %127 = getelementptr inbounds [4096 x { double, double }], ptr %2, i32 0, i32 %7
  store { double, double } %126, ptr %127, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 32}
!2 = !{i32 0, i32 128}
!3 = !{}
