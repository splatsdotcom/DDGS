import torch
import torch.nn as nn
import math
from typing import NamedTuple
from . import _C

# ------------------------------------------- #

def _look(eye, forward, up):
	f = forward / torch.norm(forward)
	u = up / torch.norm(up)

	s = torch.cross(f, u, dim=0)
	s = s / torch.norm(s)
	
	u = torch.cross(s, f, dim=0)
	
	m = torch.eye(4, dtype=torch.float32, device=eye.device)
	m[0, :3] = s
	m[1, :3] = u
	m[2, :3] = -f
	m[0, 3] = -torch.dot(s, eye)
	m[1, 3] = -torch.dot(u, eye)
	m[2, 3] = torch.dot(f, eye)

	return m

def _perspective(fovY, aspect, zNear, zFar):
	tanHalfFov = math.tan(fovY / 2)

	m = torch.zeros((4, 4), dtype=torch.float32)
	m[0, 0] = 1 / (aspect * tanHalfFov)
	m[1, 1] = 1 / tanHalfFov
	m[2, 2] = -(zFar + zNear) / (zFar - zNear)
	m[2, 3] = -(2 * zFar * zNear) / (zFar - zNear)
	m[3, 2] = -1.0
	
	return m
	
# ------------------------------------------- #

class RenderOutputs:
	COLOR = 1 << 0
	ALPHA = 1 << 1
	DEPTH = 1 << 2

class Settings:	
	def __init__(self, width: int, height: int, 
				 view: torch.Tensor, proj: torch.Tensor, focalX: float, focalY: float,
				 outputs: RenderOutputs = RenderOutputs.COLOR, debug: bool = False):

		self.outputs = outputs

		self.cSettings = torch.classes.ddgs.Settings(
			width,
			height,
			view,
			proj,
			focalX,
			focalY,
			debug
		)

	@staticmethod
	def from_colmap(width: int, height: int, focalX: float, focalY: float, R: torch.Tensor, T: torch.Tensor,
					debug: bool = False):
		
		# flip sign convention:
		# ---------------
		R = -R

		# get view matrix
		# ---------------
		camPos = -R.T @ T

		forward = R[2, :3]
		up = R[1, :3]
		
		view = _look(camPos, forward, up)

		# get proj matrix:
		# ---------------
		aspect = width / height
		fovY = 2 * math.atan(height / (2 * focalY))
		zNear = 0.1
		zFar = 1000.0
		
		proj = _perspective(fovY, aspect, zNear, zFar).to(R.device)

		focalX = width / (2 * math.tan(fovY / 2))
		focalY = focalX

		# return settings:
		# ---------------
		return Settings(
			width, height, 
			view, proj, focalX, focalY,
			debug
		)

class RenderFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, settings: Settings,
				means: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor, opacities: torch.Tensor, harmonics: torch.Tensor) -> torch.Tensor:
		
		img, alpha, depth, numRendered, geomBufs, binningBufs, imageBufs = torch.ops.ddgs.forward(
			settings.cSettings, means, scales, rotations, opacities, harmonics
		)

		ctx.numRendered = numRendered
		ctx.settings = settings
		ctx.save_for_backward(
			means, scales, rotations, opacities, harmonics,
			geomBufs, binningBufs, imageBufs
		)

		ctx.mark_non_differentiable(alpha)

		return img, alpha, depth

	@staticmethod
	def backward(ctx, grad_color, grad_alpha, grad_depth):
		means, scales, rotations, opacities, harmonics, geomBufs, binningBufs, imageBufs = ctx.saved_tensors

		dMean, dScales, dRotations, dOpacities, dHarmonics = torch.ops.ddgs.backward(
			ctx.settings.cSettings, grad_color,
			means, scales, rotations, opacities, harmonics,
			ctx.numRendered, geomBufs, binningBufs, imageBufs
		)

		return (
			None,       # settings
			dMean,      # means
			dScales,    # scales
			dRotations, # rotations
			dOpacities, # opacities
			dHarmonics  # harmonics
		)

# ------------------------------------------- #

class RenderResult(NamedTuple):
	color: torch.Tensor | None
	alpha: torch.Tensor | None
	depth: torch.Tensor | None

def render(settings: Settings,
		   means: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor, opacities: torch.Tensor, harmonics: torch.Tensor,
		   normalizeRotations=True) -> torch.Tensor:

	if normalizeRotations:
		rotations = rotations / torch.norm(rotations, dim=-1, keepdim=True).clamp(min=1e-8)

	img, alpha, depth = RenderFunction.apply(
		settings,
		means, scales, rotations, opacities, harmonics
	)

	return RenderResult(
		color=img   if settings.outputs & RenderOutputs.COLOR else None,
		alpha=alpha if settings.outputs & RenderOutputs.ALPHA else None,
		depth=depth if settings.outputs & RenderOutputs.DEPTH else None,
	)

class Renderer(nn.Module):
	def __init__(self, settings, normalizeRotations=True):
		super().__init__()
		self.settings = settings
		self.normalizeRotations = normalizeRotations

	def forward(self, means: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor, opacities: torch.Tensor, harmonics: torch.Tensor):
		return render(
			self.settings, 
			means, scales, rotations, opacities, harmonics, 
			self.normalizeRotations
		)