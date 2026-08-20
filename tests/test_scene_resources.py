"""碰撞资产字节、模型引用与Scene形状记录的严格接缝。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from physics_engine.geometry import MassProperties
from physics_engine.materials import EvidenceRef, MaterialProperty, MaterialRecord
from physics_engine.model_snapshot import AssetRole, ModelAssetRef
from physics_engine.motion import Pose
from physics_engine.scene_resources import (
    AnalyticCollisionRecord,
    CollisionAssetLoadSpec,
    MassPropertiesRecord,
    SceneResourceCatalog,
    SceneResourceError,
    load_collision_asset,
)
from physics_engine.shapes import CollisionShape, MeshAsset, Sphere


def _asset(path: Path, *, role: AssetRole = AssetRole.COLLISION) -> ModelAssetRef:
    return ModelAssetRef(
        asset_id="asset/workpiece-collision",
        role=role,
        path_relative="assets/workpiece.collision.asset",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        format="asset",
        units="mm",
        frame_id="frame/workpiece-asset",
        component_from_asset=Pose(
            (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        ),
    )


def _spec() -> CollisionAssetLoadSpec:
    return CollisionAssetLoadSpec(
        asset_id="asset/workpiece-collision",
        direction="fitted",
        convexity="nonconvex_declared",
        aabb_min_mm=(-1.0, -1.0, -1.0),
        aabb_max_mm=(1.0, 1.0, 1.0),
    )


def test_collision_asset_loader_rehashes_bytes_and_builds_an_exact_mesh_record(
    tmp_path: Path,
):
    path = tmp_path / "assets" / "workpiece.collision.asset"
    path.parent.mkdir()
    path.write_bytes(b"synthetic collision bytes\n")
    reference = _asset(path)

    loaded = load_collision_asset(tmp_path, reference, _spec())

    assert loaded.asset == reference
    assert loaded.byte_length == len(path.read_bytes())
    assert isinstance(loaded.collision.shape, MeshAsset)
    assert loaded.collision.shape.path_relative == reference.path_relative
    assert loaded.collision.shape.sha256 == reference.sha256
    assert loaded.collision.shape.usage == "collision"


def test_red_changed_asset_bytes_are_rejected_before_scene_assembly(tmp_path: Path):
    path = tmp_path / "assets" / "workpiece.collision.asset"
    path.parent.mkdir()
    path.write_bytes(b"locked bytes\n")
    reference = _asset(path)
    path.write_bytes(b"changed bytes\n")

    with pytest.raises(SceneResourceError, match="sha256"):
        load_collision_asset(tmp_path, reference, _spec())


def test_red_visual_asset_cannot_enter_the_collision_resource_loader(tmp_path: Path):
    path = tmp_path / "assets" / "workpiece.collision.asset"
    path.parent.mkdir()
    path.write_bytes(b"visual bytes\n")

    with pytest.raises(SceneResourceError, match="collision role"):
        load_collision_asset(tmp_path, _asset(path, role=AssetRole.VISUAL), _spec())


def test_red_asset_specification_must_name_the_same_asset(tmp_path: Path):
    path = tmp_path / "assets" / "workpiece.collision.asset"
    path.parent.mkdir()
    path.write_bytes(b"asset bytes\n")
    bad = CollisionAssetLoadSpec(
        asset_id="asset/another",
        direction="fitted",
        convexity="nonconvex_declared",
        aabb_min_mm=(-1.0, -1.0, -1.0),
        aabb_max_mm=(1.0, 1.0, 1.0),
    )

    with pytest.raises(SceneResourceError, match="names another asset"):
        load_collision_asset(tmp_path, _asset(path), bad)


def test_red_symlink_cannot_escape_the_declared_package_root(tmp_path: Path):
    package = tmp_path / "package"
    assets = package / "assets"
    assets.mkdir(parents=True)
    outside = tmp_path / "outside.asset"
    outside.write_bytes(b"outside bytes\n")
    link = assets / "workpiece.collision.asset"
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable on this platform: {error}")
    reference = _asset(link)

    with pytest.raises(SceneResourceError, match="escapes package_root"):
        load_collision_asset(package, reference, _spec())


def test_resource_catalog_keeps_asset_and_analytic_sources_separate(tmp_path: Path):
    path = tmp_path / "assets" / "workpiece.collision.asset"
    path.parent.mkdir()
    path.write_bytes(b"asset bytes\n")
    reference = _asset(path)
    loaded = load_collision_asset(tmp_path, reference, _spec())
    analytic = AnalyticCollisionRecord(
        shape_id="shape/guide-sphere",
        shape_frame_id="frame/guide-sphere",
        collision=CollisionShape(Sphere(5.0), "fitted"),
        component_from_shape=Pose(
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        ),
    )
    catalog = SceneResourceCatalog(
        collision_assets=(loaded,), analytic_shapes=(analytic,)
    )

    assert catalog.collision_asset(reference.asset_id) is loaded
    assert catalog.analytic_shape(analytic.shape_id) is analytic
    with pytest.raises(SceneResourceError, match="no collision asset"):
        catalog.collision_asset("asset/missing")
    with pytest.raises(SceneResourceError, match="no analytic shape"):
        catalog.analytic_shape("shape/missing")


def test_resource_catalog_resolves_sealed_material_and_named_mass_properties():
    material = MaterialRecord(
        material_id="material/workpiece",
        applicable_domains=("mechanics",),
        properties=(
            MaterialProperty(
                "density_kg_m3",
                1000.0,
                ("mechanics",),
                EvidenceRef(
                    "estimated",
                    "evidence/resource-material",
                    "Synthetic resource catalog fixture.",
                ),
            ),
        ),
    ).sealed()
    mass = MassPropertiesRecord.create(
        mass_properties_id="mass-properties/workpiece",
        geometry_resource_id="asset/workpiece-collision",
        expressed_in_frame_id="frame/workpiece-asset",
        properties=MassProperties(
            8.0,
            (0.0, 0.0, 0.0),
            1.0,
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ),
        evidence=EvidenceRef(
            "estimated",
            "evidence/resource-mass",
            "Synthetic resource mass properties.",
        ),
    )
    catalog = SceneResourceCatalog(
        collision_assets=(),
        analytic_shapes=(),
        materials=(material,),
        mass_property_records=(mass,),
    )
    assert catalog.material(material.material_id) is material
    assert catalog.mass_properties(mass.mass_properties_id) is mass
