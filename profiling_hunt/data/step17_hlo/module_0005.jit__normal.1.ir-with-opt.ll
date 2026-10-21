; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_add_fusion(ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %0, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(8) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = load <2 x i32>, ptr addrspace(1) %3, align 16, !invariant.load !3
  %7 = extractelement <2 x i32> %6, i32 0
  %8 = extractelement <2 x i32> %6, i32 1
  %9 = add i32 %8, %5
  %10 = tail call i32 @llvm.fshl.i32(i32 %9, i32 %9, i32 13)
  %11 = add i32 %7, %9
  %12 = xor i32 %11, %10
  %13 = tail call i32 @llvm.fshl.i32(i32 %12, i32 %12, i32 15)
  %14 = add i32 %12, %11
  %15 = xor i32 %14, %13
  %16 = add i32 %15, %14
  %17 = tail call i32 @llvm.fshl.i32(i32 %15, i32 %15, i32 26)
  %18 = xor i32 %16, %17
  %19 = tail call i32 @llvm.fshl.i32(i32 %18, i32 %18, i32 6)
  %20 = add i32 %18, %16
  %21 = xor i32 %20, %19
  %22 = xor i32 %8, %7
  %23 = xor i32 %22, 466688986
  %24 = add i32 %23, 1
  %25 = add i32 %24, %21
  %26 = add i32 %20, %8
  %27 = tail call i32 @llvm.fshl.i32(i32 %25, i32 %25, i32 17)
  %28 = add i32 %26, %25
  %29 = xor i32 %28, %27
  %30 = tail call i32 @llvm.fshl.i32(i32 %29, i32 %29, i32 29)
  %31 = add i32 %29, %28
  %32 = xor i32 %31, %30
  %33 = add i32 %32, %31
  %34 = tail call i32 @llvm.fshl.i32(i32 %32, i32 %32, i32 16)
  %35 = xor i32 %33, %34
  %36 = tail call i32 @llvm.fshl.i32(i32 %35, i32 %35, i32 24)
  %37 = add i32 %35, %33
  %38 = xor i32 %37, %36
  %39 = add i32 %7, 2
  %40 = add i32 %39, %38
  %41 = add i32 %37, %23
  %42 = tail call i32 @llvm.fshl.i32(i32 %40, i32 %40, i32 13)
  %43 = add i32 %41, %40
  %44 = xor i32 %43, %42
  %45 = tail call i32 @llvm.fshl.i32(i32 %44, i32 %44, i32 15)
  %46 = add i32 %44, %43
  %47 = xor i32 %46, %45
  %48 = add i32 %47, %46
  %49 = tail call i32 @llvm.fshl.i32(i32 %47, i32 %47, i32 26)
  %50 = xor i32 %48, %49
  %51 = tail call i32 @llvm.fshl.i32(i32 %50, i32 %50, i32 6)
  %52 = add i32 %50, %48
  %53 = xor i32 %52, %51
  %54 = add i32 %8, 3
  %55 = add i32 %54, %53
  %56 = add i32 %52, %7
  %57 = tail call i32 @llvm.fshl.i32(i32 %55, i32 %55, i32 17)
  %58 = add i32 %56, %55
  %59 = xor i32 %58, %57
  %60 = tail call i32 @llvm.fshl.i32(i32 %59, i32 %59, i32 29)
  %61 = add i32 %59, %58
  %62 = xor i32 %61, %60
  %63 = add i32 %62, %61
  %64 = tail call i32 @llvm.fshl.i32(i32 %62, i32 %62, i32 16)
  %65 = xor i32 %63, %64
  %66 = tail call i32 @llvm.fshl.i32(i32 %65, i32 %65, i32 24)
  %67 = add i32 %65, %63
  %68 = xor i32 %67, %66
  %69 = add i32 %23, 4
  %70 = add i32 %69, %68
  %71 = add i32 %67, %8
  %72 = tail call i32 @llvm.fshl.i32(i32 %70, i32 %70, i32 13)
  %73 = add i32 %71, %70
  %74 = xor i32 %73, %72
  %75 = tail call i32 @llvm.fshl.i32(i32 %74, i32 %74, i32 15)
  %76 = add i32 %74, %73
  %77 = xor i32 %76, %75
  %78 = tail call i32 @llvm.fshl.i32(i32 %77, i32 %77, i32 26)
  %79 = add i32 %77, %76
  %80 = xor i32 %79, %78
  %81 = add i32 %79, %23
  %82 = add i32 %81, %80
  %83 = zext nneg i32 %5 to i64
  %84 = getelementptr inbounds [4 x i8], ptr addrspace(1) %4, i64 %83
  store i32 %82, ptr addrspace(1) %84, align 4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.fshl.i32(i32, i32, i32) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="2,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 2}
!3 = !{}
