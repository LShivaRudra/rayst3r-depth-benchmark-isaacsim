import omni.usd
from pxr import UsdGeom, Gf

stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath("/World/textured")
if not prim:
  print("Prim not found at /World/textured")
else:
  bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
  r = bbox_cache.ComputeWorldBound(prim).GetRange()
  sz = r.GetSize()
  print("size (m):", sz)

  max_dim = max(sz[0], sz[1], sz[2])
  if max_dim < 0.01:   # likely mm-scale
      x = UsdGeom.Xformable(prim)
      x.AddScaleOp().Set(Gf.Vec3f(1000,1000,1000))
      print("Applied x1000 scale")
  elif max_dim < 0.1:  # likely cm-scale
      x = UsdGeom.Xformable(prim)
      x.AddScaleOp().Set(Gf.Vec3f(100,100,100))
      print("Applied x100 scale")

