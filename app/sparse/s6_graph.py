"""S6 compiler rewrites; standard ONNX only, no custom CPU operations."""
import math
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
def lower_broadcasts(m):
    m=onnx.shape_inference.infer_shapes(m)
    sh={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(m.graph.value_info)+list(m.graph.input)+list(m.graph.output)};sh.update({v.name:list(v.dims) for v in m.graph.initializer});const={v.name:nh.to_array(v) for v in m.graph.initializer};new=[]
    def c(name,val):m.graph.initializer.append(nh.from_array(np.asarray(val),name));return name
    def op(typ,inputs,name,out=None,**attr):
     value=out or name+'_out';new.append(h.make_node(typ,inputs,[value],name=name,**attr));return value
    def rs(x,shape,name):return op('Reshape',[x,c(name+'_shape',np.array(shape,np.int64))],name)
    consumers={}
    for v in m.graph.node:
     for x in v.input:consumers.setdefault(x,[]).append(v)
    skip=set()
    for n in m.graph.node:
     if n.name in skip:continue
     b=n.name+'_lower';a=sh.get(n.input[0],[]) if n.input else []
     if n.op_type=='MatMul' and len(a)==4 and a[0:2]==[1,6] and a[-1]==4 and len(sh.get(n.input[1],[]))==2:
      rows=a[-2];count=sh[n.input[1]][-1]
      matrix=rs(n.input[0],[6,rows,4],b+'_a')
      matrix=op('Transpose',[matrix],b+'_at',perm=[2,0,1]);matrix=rs(matrix,[4,6*rows],b+'_af')
      points=op('Transpose',[n.input[1]],b+'_points',perm=[1,0])
      product=op('MatMul',[points,matrix],b+'_mm')
      packed=rs(product,[count,6,rows],b+'_packed')
      packed=op('Transpose',[packed],b+'_transposed',perm=[1,2,0])
      op('Reshape',[packed,c(b+'_osh',np.array(sh[n.output[0]],np.int64))],b+'_out',n.output[0]);continue
     if n.op_type=='MatMul' and len(a)>=6 and a[0:2]==[1,6] and a[-1]==4:
      bs=sh[n.input[1]];rows=a[-2];spatial=bs[2:-2]
      matrix=n.input[0];outname=n.output[0];outshape=sh[outname]
      ar=rs(matrix,[6,rows,4],b+'_a');at=op('Transpose',[ar],b+'_at',perm=[2,0,1]);af=rs(at,[4,6*rows],b+'_af')
      bf=rs(n.input[1],[-1,4],b+'_bf');prod=op('MatMul',[bf,af],b+'_mm')
      packed=rs(prod,list(spatial)+[6,rows],b+'_packed')
      k=len(spatial);trans=op('Transpose',[packed],b+'_t',perm=[k]+list(range(k))+[k+1])
      op('Reshape',[trans,c(b+'_osh',np.array(outshape,np.int64))],b+'_out',outname);continue
     if n.op_type in ['ReduceSum','ReduceMean']:
      attr={v.name:h.get_attribute_value(v) for v in n.attribute};axes=const.get(n.input[1]) if len(n.input)>1 else attr.get('axes');osh=sh.get(n.output[0])
      if axes is not None and a and osh and all(osh) and all(a[int(i)]==1 for i in axes):
       op('Reshape',[n.input[0],c(b+'_shape',np.array(osh,np.int64))],b,n.output[0]);continue
     if n.op_type=='Mul' and len(sh.get(n.output[0],[]))>=6:
      osh=sh[n.output[0]];rank=len(osh);ish=[sh[v] for v in n.input];ish=[[1]*(rank-len(v))+v for v in ish];groups=[];prev=None
      for i in range(rank):
       pattern=tuple(v[i]==1 for v in ish)
       if pattern==prev:groups[-1].append(i)
       else:groups.append([i]);prev=pattern
      if len(groups)<rank:
       shapes=[[math.prod(v[i] for i in g) for g in groups] for v in ish]
       v=[rs(x,shape,b+'_'+str(i)) for i,(x,shape) in enumerate(zip(n.input,shapes))]
       out=op('Mul',v,b+'_mul');op('Reshape',[out,c(b+'_osh',np.array(osh,np.int64))],b+'_out',n.output[0]);continue
     new.append(n)
    del m.graph.node[:];m.graph.node.extend(new);del m.graph.value_info[:];needed={v.name for v in m.graph.output};keep=[]
    for n in reversed(m.graph.node):
     if any(v in needed for v in n.output):keep.append(n);needed.update(n.input)
    del m.graph.node[:];m.graph.node.extend(keep[::-1]);inits=[v for v in m.graph.initializer if v.name in needed];del m.graph.initializer[:];m.graph.initializer.extend(inits)
    onnx.checker.check_model(m)
    return m



def avoid_qat_pattern(model):
    """Avoid SDK Round->Clip->Mul QAT matcher, respecting optional Clip bounds."""
    model=onnx.shape_inference.infer_shapes(model)
    rank={v.name:len(v.type.tensor_type.shape.dim) for v in model.graph.value_info}
    const={v.name:nh.to_array(v) for v in model.graph.initializer}
    nodes=[];changed=0
    for n in model.graph.node:
        if n.op_type!='Clip' or not n.name.startswith('/occ_branch/view_transform/'):
            nodes.append(n);continue
        bounds=[(i,x) for i,x in enumerate(n.input[1:]) if x]
        current=n.input[0]
        for j,(i,x) in enumerate(bounds):
            key=n.name+'_bound_'+str(i);value=const[x]
            model.graph.initializer.append(nh.from_array(np.full([1]*rank[n.input[0]],value.item(),dtype=value.dtype),key))
            out=n.output[0] if j==len(bounds)-1 else n.name+'_lower_bound'
            nodes.append(h.make_node('Max' if i==0 else 'Min',[current,key],[out],name=n.name+'_bound_op_'+str(i)))
            current=out
        changed+=1
    del model.graph.node[:];model.graph.node.extend(nodes)
    return model,changed


def geometry_precision_tensors(model):
    """Keep geometric coordinates in FP16, masks and learned feature maps INT8."""
    m=onnx.shape_inference.infer_shapes(model)
    values={v.name:v for v in list(m.graph.input)+list(m.graph.output)+list(m.graph.value_info)}
    typ={k:v.type.tensor_type.elem_type for k,v in values.items()}
    shapes={k:[d.dim_value for d in v.type.tensor_type.shape.dim] for k,v in values.items()}
    prod={v:n for n in m.graph.node for v in n.output}
    deps={v.name:{v.name} for v in m.graph.input}
    deps.update({v.name:{v.name} for v in m.graph.initializer if v.name in ('projection_mat','occ_ego2img')}); bool_deps={'history_valid'}
    constants={v.name:nh.to_array(v) for v in m.graph.initializer}
    mask_constants={k for k,v in constants.items() if np.all((v==0)|(v==1))}
    mask_views={'Cast','Reshape','Transpose','Unsqueeze','Squeeze','Slice','Gather','Expand','Identity'}
    for n in m.graph.node:
        d=set().union(*(deps.get(x,set()) for x in n.input))
        for x in n.output:
            deps[x]=d
            if typ.get(x)==9 or n.name.endswith('_numeric_mask_ceil'):
                bool_deps.add(x)
            elif n.op_type in mask_views and n.input[0] in bool_deps:
                bool_deps.add(x)
            elif n.op_type in {'Mul','Add','Sub','Min','Max'} and all(y in bool_deps or y in mask_constants for y in n.input):
                bool_deps.add(x)
    chosen={'projection_mat','occ_ego2img','history_anchor','history_transform','det_boxes','next_history_anchor'}
    views={'Reshape','Transpose','Unsqueeze','Squeeze','Slice','Gather','Identity'}
    for n in m.graph.node:
        for x in n.output:
            if typ.get(x)!=1 or x in bool_deps:continue
            if deps.get(x)=={'occ_ego2img'}:chosen.add(x)
            if '/kps_generator/' in n.name:chosen.add(x)
            if n.name.startswith('/det_head/') and shapes.get(x,[])[-1:]==[11]:chosen.add(x)
            if n.op_type in views and deps.get(x) in ({'projection_mat'},{'history_transform'}) and math.prod(shapes.get(x,[1000]))<=96:chosen.add(x)
    # Trace only the detection projection arithmetic; stop before the learned point generator.
    for n in m.graph.node:
        if n.op_type!='GridSample' or not n.name.startswith('/det_head/'):continue
        seen=set()
        def visit(x):
            if x in seen:return
            seen.add(x);p=prod.get(x)
            if typ.get(x)==1 and x not in bool_deps:chosen.add(x)
            if p is None or '/kps_generator/' in p.name:return
            for y in p.input:
                if y in prod:visit(y)
                elif y in values and typ.get(y)==1:chosen.add(y)
        visit(n.input[1])
    for n in m.graph.node:
        if not any(x in chosen for x in n.output):continue
        if n.name.startswith('/occ_branch/view_transform/') or '_lower_mm' in n.name:
            for x in n.input:
                if x in constants and constants[x].dtype==np.float32:
                    chosen.add(x);typ[x]=1
    def constant_views(x):
        if x in constants:
            if constants[x].dtype==np.float32:chosen.add(x);typ[x]=1
            return
        producer=prod.get(x)
        if producer is not None and producer.op_type in views and not deps.get(x):
            if typ.get(x)==1:chosen.add(x)
            constant_views(producer.input[0])
    for node in m.graph.node:
        if node.op_type=='MatMul' and any(x in chosen for x in node.output):
            for x in node.input:constant_views(x)
    return sorted(x for x in chosen if typ.get(x)==1)


def numeric_comparison_masks(model):
    """Replace compare->float Cast by exact 0/1 arithmetic for finite values.

    The SDK lists FP16 comparisons as supported but their bool/Cast backend
    combination cannot be serialized. ceil(min(max(x-threshold,0),1)) equals
    float(x>threshold), including values arbitrarily close to the threshold.
    Bool values consumed by Where/other logical operators are left unchanged.
    """
    consumers={}
    for n in model.graph.node:
        for x in n.input:consumers.setdefault(x,[]).append(n)
    zero='s6_mask_zero';one='s6_mask_one'
    model.graph.initializer.extend([nh.from_array(np.array(0,np.float32),zero),nh.from_array(np.array(1,np.float32),one)])
    nodes=[];skip=set();changed=0
    for n in model.graph.node:
        if n.name in skip:continue
        uses=consumers.get(n.output[0],[])
        if n.op_type not in ('Greater','Less') or not uses or not all(u.op_type=='Cast' and any(a.name=='to' and a.i==1 for a in u.attribute) for u in uses):
            nodes.append(n);continue
        stem=n.name+'_numeric_mask';a,b=n.input
        if n.op_type=='Less':a,b=b,a
        nodes.extend([h.make_node('Sub',[a,b],[stem+'_difference'],name=stem+'_difference'),
                      h.make_node('Max',[stem+'_difference',zero],[stem+'_low'],name=stem+'_low'),
                      h.make_node('Min',[stem+'_low',one],[stem+'_clip'],name=stem+'_clip'),
                      h.make_node('Ceil',[stem+'_clip'],[stem+'_value'],name=stem+'_ceil')])
        for u in uses:
            nodes.append(h.make_node('Identity',[stem+'_value'],list(u.output),name=u.name))
            skip.add(u.name)
        changed+=1
    del model.graph.node[:];model.graph.node.extend(nodes);del model.graph.value_info[:]
    return model,changed
